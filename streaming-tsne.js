const math = require('mathjs')
const gaussian = require('gaussian')
const ndarray = require('ndarray')
const pool = require('ndarray-scratch')
const ops = require("ndarray-ops")
const ndtest = require('ndarray-tests');

var tsne = tsne || {}

;(function (global) {
  'use strict'

  var seed = 1
  const tmpRandom = function () {
    const x = Math.sin(seed++) * 10000
    return x - Math.floor(x)
  }

  const initialY = function (numSamples) {
    const distribution = gaussian(0, 1e-4)
    const ys = pool.zeros([numSamples, 2])
    for (let i = 0; i < numSamples; i++) {
      ys.set(i, 0, distribution.ppf(tmpRandom()))
      ys.set(distribution.ppf(tmpRandom()))
    }
    return ys
  }

  const lowDimAffinities = function (y) {
    let q = math.zeros([y.length, y.length])
    for (let i = 0; i < y.length; i++) {
      for (let j = i + 1; j < y.length; j++) {
        const diff = math.subtract(y[i], y[j])
        const affinity = 1.0 / (1 + math.dot(diff, diff))
        q[i][j] = q[j][i] = math.max(affinity, 1e-12)
      }
    }
    q = math.divide(q, math.sum(q))
    return q
  }

  const gradKL = function (p, q, y) {
    let gradTot = math.zeros([y.length, y[0].length])
    for (let i = 0; i < y.length; i++) {
      for (let d = 0; d < y[0].length; d++) {
        let acc = 0
        for (let j = 0; j < y.length; j++) {
          if (i === j) continue
          let diff = y[i][d] - y[j][d]
          let component = diff * (p[i][j] - q[i][j]) / (1 + (diff * diff))
          acc += component
        }
        gradTot[i][d] = 4 * acc
      }
    }
    return gradTot
  }

  const updateY = function (grad, ytMinus1, ytMinus2) {
    const learningRate = 100
    const alpha = 0.5
    const yT = math.add(math.add(ytMinus1, math.multiply(learningRate, grad)), math.multiply(alpha, math.subtract(ytMinus1, ytMinus2)))
    return yT
  }

  const computeCost = function (p, q) {
    var cost = 0
    for (var i = 0; i < math.size(p)[0]; i++) {
      for (var j = 0; j < math.size(p)[0]; j++) {
        if (i === j) continue
        cost += p[i][j] * Math.log(p[i][j] / q[i][j])
      }
    }
    return cost
  }

  const euclideanDistance = function (x, y) {
    // Compute Euclidean distance
    const m = x.length
    let d = 0
    for (let k = 0; k < m; k++) {
      let xk = x[k]
      let yk = y[k]
      d += (xk - yk) * (xk - yk)
    }
    return d
  }

  // Compute pairwise Euclidean distances.
  // NOT Based on whole squared expansion:
  // (a - b)^2 = a^2 - 2ab + b^2, where a and b are rows of x.
  // CURRENTLY brute force until we reach correctness and can optimize.
  const XToD = function (X) {
    // const n = X.shape[0]
    // const m = X.shape[1]
    // X is an array of arrays
    const n = X.length
    const D = pool.zeros([n, n])
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const d = euclideanDistance(X[i], X[j])
        D.set(i, j, d)
        D.set(j, i, d)
      }
    }
    return D
  }

  const getpIAndH = function (Pi, Di, beta, i) {
    // Compute a single row Pi of the kernel and the Shannon entropy H
    // FIXME: can reuse this array for subsequent calls
    const m = Di.shape[0]
    ops.muls(Pi, Di, -beta)  // scalar multiply by -beta, store in Pi
    ops.expeq(Pi)             // exponentiate Pi in place
    Pi.set(i, 0)              // affinity is zero between the same point

    // Normalize
    const sumPi = ops.sum(Pi)
    if (sumPi === 0) {
      ops.assigns(sumPi, 0)
    } else {
      ops.divseq(Pi, sumPi)
    }

    // Compute entropy H
    // FIXME: this can be vectorized too, but maybe slower
    let H = 0
    for (var j = 0; j < m; j++) {
      const Pji = Pi.get(j)
      // Skip small values to avoid NaNs or exploding values
      if (Pji > 1e-7) {
        H -= Pji * Math.log2(Pji)
      }
    }

    return H
  }

  const symmetrize = function (P) {
    // Symmetrize in place according to:
    //         p_j|i + p_i|j
    // p_ij = ---------------
    //              2n
    const n = P.shape[0]
    ops.addeq(P, P.transpose(1, 0))
    ops.divseq(P, 2 * n)
  }

  const DToP = function (D, perplexity) {
    const n = D.shape[0]
    const P = pool.zeros([n, n])

    // Shannon entropy H is log2 of perplexity
    const Hdesired = Math.log2(perplexity)

    for (let i = 0; i < n; i++) {
      // We perform binary search to find the beta such that
      // the conditional distribution P_i has the given perplexity.
      // We define:
      //   beta = 1 / (2 * sigma_i^2)
      // where sigma_i is the bandwith of the Gaussian kernel
      // for the conditional distribution P_i
      let beta = 1
      let betamin = -Infinity
      let betamax = Infinity
      let Hdiff
      let numTries = 0
      do {
        numTries++
        const Pi = P.pick(i, null)
        const Di = D.pick(i, null)
        const H = getpIAndH(Pi, Di, beta, i)
        Hdiff = H - Hdesired

        if (Hdiff > 0) {
          // Entropy too high, beta is too small
          betamin = beta
          if (betamax === Infinity) {
            beta = beta * 2
          } else {
            beta = (beta + betamax) / 2
          }
        } else {
          // Entropy is too low, beta is too big
          betamax = beta
          if (betamin === -Infinity) {
            beta = beta / 2
          } else {
            beta = (beta + betamin) / 2
          }
        }
      } while (Math.abs(Hdiff) > 1e-05 && numTries < 50)
    }
    // console.log('Sigma mean: ' + math.mean(math.sqrt(math.dotDivide(1, beta))))
    //
    // console.log('nan?', !ndtest.equal(P, P))

    // Symmetrize conditional distribution
    symmetrize(P)
    // console.log('nan symmetrized?', !ndtest.equal(P, P))
    return P
  }

  const debugGrad = function (grad, cost, y, p) {
    var epsilon = 1e-5
    for (var i = 0; i < math.size(grad)[0]; i++) {
      for (var d = 0; d < math.size(grad)[1]; d++) {
        var yold = y[i][d]

        y[i][d] = yold + epsilon
        const q0 = lowDimAffinities(y)
        const cg0 = computeCost(p, q0)

        y[i][d] = yold - epsilon
        const q1 = lowDimAffinities(y)
        const cg1 = computeCost(p, q1)

        var analytic = grad[i][d]
        var numerical = (cg0 - cg1) / (2 * epsilon)
        if (analytic - numerical > 1e-5) {
          console.log(i + ',' + d + ': gradcheck analytic: ' + analytic + ' vs. numerical: ' + numerical)
        }
        y[i][d] = yold
      }
    }
  }

  let TSNE = function (opt) {}

  TSNE.prototype = {
    initData: function (data) {
      const numSamples = data.length
      const D = XToD(data)
      this.p = DToP(D, 30)
      this.y = initialY(numSamples)
      this.ytMinus1 = this.y
      this.ytMinus2 = this.y
    },
    step: function () {
      const q = lowDimAffinities(this.y)
      const cost = computeCost(this.p, q)
      console.log('Cost: ' + cost)
      let grad = gradKL(this.p, q, this.y)
      // debugGrad(grad, cost, this.y, this.p)
      this.yTMinus2 = this.ytMinus1
      this.ytMinus1 = this.y
      this.y = updateY(grad, this.ytMinus1, this.yTMinus2)
    }
  }

  global.TSNE = TSNE
})(tsne)

// export the library to window, or to module in nodejs
// Webpack supports both.
;(function (lib) {
  'use strict'
  if (typeof module !== 'undefined' && typeof module.exports === 'undefined') {
    module.exports = lib // in nodejs
  }
  if (typeof window !== 'undefined') {
    window.tsne = lib // in ordinary browser attach library to window
  }
})(tsne)

