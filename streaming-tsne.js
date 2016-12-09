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

  const profileTime = {}
  const profileCount = {}
  const profile = function (name, fn) {
    const tic = performance.now()
    const result = fn()
    const toc = performance.now()
    const elapsed = Math.round(toc - tic)
    if (!profileCount.hasOwnProperty(name)) {
      profileCount[name] = 0
      profileTime[name] = 0
    }
    profileCount[name]++
    profileTime[name] += (elapsed - profileTime[name]) / profileCount[name]
    console.log(name + ': ' + elapsed + 'ms' + ', avg ' + Math.round(profileTime[name]) + 'ms')
    return result
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

    // Symmetrize conditional distribution
    symmetrize(P)
    return P
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
    },

    updateQ: function () {
      // Update low dimensional affinities of the embedding
      const n = this.Y.shape[0]
      const dims = this.Y.shape[1]
      let qtotal = 0
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          let dist = 0
          for (let d = 0; d < dims; d++) {
            const diff = this.Y.get(i, d) - this.Y.get(j, d)
            dist += diff * diff
          }
          const affinity = 1. / (1. + dist)
          this.Qu.set(i, j, affinity)
          this.Qu.set(j, i, affinity)
          qtotal += 2 * affinity
        }
      }

      // Normalize
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const Qij = Math.max(this.Qu.get(i, j) / qtotal, 1e-100)
          this.Q.set(i, j, Qij)
          this.Q.set(j, i, Qij)
        }
      }
      for (let i = 0; i < n; i++) {
        this.Q.set(i, i, 1e-100)
      }
    },

    checkGrad: function () {
      const n = this.Y.shape[0]
      const dims = this.Y.shape[1]
      const epsilon = 1e-5
      for (var i = 0; i < n; i++) {
        for (var d = 0; d < dims; d++) {
          var yold = this.Y.get(i, d)

          this.Y.set(i, d, yold + epsilon)
          this.updateQ()
          const cg0 = computeKL(this.P, this.Q)

          this.Y.set(i, d, yold - epsilon)
          this.updateQ()
          const cg1 = computeKL(this.P, this.Q)

          var analytic = this.grad.get(i, d)
          var numerical = (cg0 - cg1) / (2 * epsilon)
          if (analytic - numerical > 1e-5) {
            console.error(i + ',' + d + ': analytic: ' + analytic + ' vs. numerical: ' + numerical)
          } else {
            console.log(i + ',' + d + ': analytic: ' + analytic + ' vs. numerical: ' + numerical)
          }
          this.Y.set(i, d, yold)
        }
      }
    },

    updateGrad: function () {
      profile('updateQ', () => {
        this.updateQ()
      })

      // Early exaggeration
      const exag = this.iter < 100 ? 4 : 1

      let KL = 0
      profile('computeGrad', () => {
        // Compute gradient of the KL divergence
        const n = this.Y.shape[0]
        const dims = this.Y.shape[1]
        let gradi = [0, 0]  // FIXME: 2D only
        for (let i = 0; i < n; i++) {
          // Reset
          gradi[0] = gradi[1] = 0  // FIXME: 2D only

          // Accumulate gradient over j
          for (let j = 0; j < n; j++) {
            const Pij = this.P.get(i, j)
            const Qij = this.Q.get(i, j)

            // Accumulate KL divergence
            KL -= Pij * Math.log(Qij)

            const mulFactor = 4 * (exag * Pij - Qij) * this.Qu.get(i, j)
            // Unfurled loop, but 2D only
            gradi[0] += mulFactor * (this.Y.get(i, 0) - this.Y.get(j, 0))
            gradi[1] += mulFactor * (this.Y.get(i, 1) - this.Y.get(j, 1))
          }

          // Set gradient
          for (let d = 0; d < dims; d++) {
            this.grad.set(i, d, gradi[d])
          }
        }
      })
      return KL
    },

    initData: function (data) {
      const numSamples = data.length
      const D = XToD(data)
      this.P = DToP(D, 30)
      pool.free(D)
      this.Y = initialY(numSamples)
      this.ytMinus1 = pool.clone(this.Y)
      this.ytMinus2 = pool.clone(this.Y)
      this.Ygains = pool.ones(this.Y.shape)
      this.iter = 0
      this.learningRate = 10
      this.grad = pool.zeros(this.Y.shape)
      this.Q = pool.zeros(this.P.shape)
      this.Qu = pool.zeros(this.P.shape)
    },

    step: function () {
      // Compute gradient
      const cost = this.updateGrad()
      // if (this.iter > 100) {
      //   this.checkGrad()
      // }

      // Rotate buffers
      const temp = this.ytMinus2
      this.ytMinus2 = this.ytMinus1
      this.ytMinus1 = this.Y
      this.Y = temp

      // Perform update
      profile('updateY', () => {
        this.updateY()
      })

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

