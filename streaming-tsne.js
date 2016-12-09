const math = require('mathjs')
const gaussian = require('gaussian')
const ndarray = require('ndarray')
const pool = require('ndarray-scratch')
const ops = require("ndarray-ops")
const ndtest = require('ndarray-tests');

var tsne = tsne || {}

;(function (global) {
  'use strict'

  const hasNaN = function (M) {
    return !ndtest.equal(M, M)
  }

  var seed = 1
  const tmpRandom = function () {
    const x = Math.sin(seed++) * 10000
    return x - Math.floor(x)
  }

  const initialY = function (numSamples) {
    // FIXME: allow arbitrary dimensions??
    const distribution = gaussian(0, 1e-4)
    const ys = pool.zeros([numSamples, 2])
    for (let i = 0; i < numSamples; i++) {
      ys.set(i, 0, distribution.ppf(Math.random()))
      ys.set(i, 1, distribution.ppf(Math.random()))
    }
    return ys
  }

  const lowDimAffinities = function (Y) {
    // FIXME: Q and Qu need to be freed
    const n = Y.shape[0]
    const dims = Y.shape[1]
    const Qu = pool.zeros([n, n])
    let qtotal = 0
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dist = 0
        for (let d = 0; d < dims; d++) {
          const diff = Y.get(i, d) - Y.get(j, d)
          dist += diff * diff
        }
        const affinity = 1. / (1. + dist)
        Qu.set(i, j, affinity)
        Qu.set(j, i, affinity)
        qtotal += 2 * affinity
      }
    }
    const Q = pool.zeros([n, n])
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const Qij = Math.max(Qu.get(i, j) / qtotal, 1e-100)
        Q.set(i, j, Qij)
        Q.set(j, i, Qij)
      }
    }
    for (let i = 0; i < n; i++) {
      Q.set(i, i, 1e-100)
    }
    return [Q, Qu]
  }

  const gradKL = function (grad, P, Y, iter) {
    const [Q, Qu] = lowDimAffinities(Y)

    // Early exaggeration
    const exag = iter < 100 ? 4 : 1;

    // Compute gradient of the KL divergence
    const n = Y.shape[0]
    const dims = Y.shape[1]
    let KL = 0
    let gradi = [0, 0]  // FIXME: 2D only
    for (let i = 0; i < n; i++) {
      // Reset
      gradi[0] = gradi[1] = 0  // FIXME: 2D only

      // Accumulate gradient over j
      for (let j = 0; j < n; j++) {

        // Accumulate KL divergence
        KL -= P.get(i, j) * Math.log(Q.get(i, j))

        const mulFactor = 4 * (exag * P.get(i, j) - Q.get(i, j)) * Qu.get(i, j)
        for (let d = 0; d < dims; d++) {
          gradi[d] += mulFactor * (Y.get(i, d) - Y.get(j, d))
        }
      }
      // Set gradient
      for (let d = 0; d < dims; d++) {
        grad.set(i, d, gradi[d])
      }
    }
    pool.free(Q)
    pool.free(Qu)
    return KL
  }

  const computeKL = function (P, Q) {
    // Compute KL divergence, minus the constant term of sum(p_ij * log(p_ij))
    const n = P.shape[0]
    let cost = 0
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        cost -= P.get(i, j) * Math.log(Q.get(i, j))
      }
    }
    return cost
  }

  const euclideanDistance = function (x, y) {
    // Compute Euclidean distance between two vectors as Arrays
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
    const n = Di.shape[0]
    //for (let j = 0; j < n; j++) {
    //  Pi.set(j, Math.exp(- beta * Di))
    //}

    ops.muls(Pi, Di, -beta)   // scalar multiply by -beta, store in Pi
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
    let H = 0
    for (var j = 0; j < n; j++) {
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
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const Pij = (P.get(i, j) + P.get(j, i)) / (2 * n)
        P.set(i, j, Pij)
        P.set(j, i, Pij)
      }
    }
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

  const checkGrad = function (grad, cost, Y, p) {
    const n = Y.shape[0]
    const dims = Y.shape[1]
    const epsilon = 1e-5
    for (var i = 0; i < n; i++) {
      for (var d = 0; d < dims; d++) {
        var yold = Y.get(i, d)

        Y.set(i, d, yold + epsilon)
        const [q0, qu0] = lowDimAffinities(Y)
        const cg0 = computeKL(p, q0)
        pool.free(q0)
        pool.free(qu0)

        Y.set(i, d, yold - epsilon)
        const [q1, qu1] = lowDimAffinities(Y)
        const cg1 = computeKL(p, q1)
        pool.free(q1)
        pool.free(qu1)

        var analytic = grad.get(i, d)
        var numerical = (cg0 - cg1) / (2 * epsilon)
        if (analytic - numerical > 1e-5) {
          console.error(i + ',' + d + ': analytic: ' + analytic + ' vs. numerical: ' + numerical)
        } else {
          console.log(i + ',' + d + ': analytic: ' + analytic + ' vs. numerical: ' + numerical)
        }
        Y.set(i, d, yold)
      }
    }
  }

  function sign(x) { return x > 0 ? 1 : x < 0 ? -1 : 0; }

  let TSNE = function (opt) {}

  TSNE.prototype = {
    updateY: function () {
      // Perform gradient update in place
      const alpha = this.iter < 250 ? 0.5 : 0.8
      const n = this.Y.shape[0]
      const dims = this.Y.shape[1]
      let Ymean = [0, 0]  // FIXME: only two dimensional
      for (let i = 0; i < n; i++) {
        for (let d = 0; d < dims; d++) {
          const gradid = this.grad.get(i, d)
          const stepid = this.ytMinus1.get(i, d) - this.ytMinus2.get(i, d)
          const gainid = this.Ygains.get(i, d)

          // Update gain
          const newgain = Math.max(
              sign(gradid) === sign(stepid) ? gainid * 0.8 : gainid + 0.2, 0.01)
          this.Ygains.set(i, d, newgain)

          // Update Y
          const Yid = (this.ytMinus1.get(i, d)
                       - this.learningRate * newgain * gradid
                       + alpha * stepid)
          this.Y.set(i, d, Yid)

          // Accumulate mean for centering
          Ymean[d] += Yid
        }
      }

      // Recenter
      for (let i = 0; i < n; i++) {
        for (let d = 0; d < dims; d++) {
          this.Y.set(i, d, this.Y.get(i, d) - Ymean[d] / n)
        }
      }
      this.learningRate *= 0.99
    },

    initData: function (data) {
      const numSamples = data.length
      const D = XToD(data)
      this.p = DToP(D, 30)
      this.Y = initialY(numSamples)
      this.ytMinus1 = pool.clone(this.Y)
      this.ytMinus2 = pool.clone(this.Y)
      this.Ygains = pool.ones(this.Y.shape)
      this.iter = 0
      this.learningRate = 10
      this.grad = pool.zeros(this.Y.shape)
    },

    step: function () {
      const cost = gradKL(this.grad, this.p, this.Y, this.iter)
      // if (this.iter == 10) {
      //   checkGrad(grad, cost, this.Y, this.p)
      // }
      const temp = this.ytMinus2
      this.ytMinus2 = this.ytMinus1
      this.ytMinus1 = this.Y
      this.Y = temp  // recycle buffer
      this.updateY()
      this.iter++
      return cost
    },
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

