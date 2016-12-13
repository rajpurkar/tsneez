var gaussian = require('gaussian')
var pool = require('ndarray-scratch')
var bhtree = require('./bhtree.js')
var ops = require('ndarray-ops')
var vptree = require('./vptree.js')
var tsne = tsne || {}

;(function (global) {
  'use strict'

  var initialY = function (numSamples) {
    // FIXME: allow arbitrary dimensions??
    var distribution = gaussian(0, 1e-4)
    var ys = pool.zeros([numSamples, 2])
    for (var i = 0; i < numSamples; i++) {
      ys.set(i, 0, distribution.ppf(Math.random()))
      ys.set(i, 1, distribution.ppf(Math.random()))
    }
    return ys
  }

  var computeKL = function (P, Q) {
    // Compute KL divergence, minus the constant term of sum(p_ij * log(p_ij))
    var n = P.shape[0]
    var cost = 0
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        cost -= P.get(i, j) * Math.log(Q.get(i, j))
      }
    }
    return cost
  }

  var squaredEuclidean = function (x, y) {
    // Compute Euclidean distance between two vectors as Arrays
    var m = x.length
    var d = 0
    for (var k = 0; k < m; k++) {
      var xk = x[k]
      var yk = y[k]
      d += (xk - yk) * (xk - yk)
    }
    return d
  }

<<<<<<< HEAD
  var euclidean = function (x, y) {
    return Math.sqrt(squaredEuclidean(x, y))
  }

  var getpIAndH = function (Pi, Xi, beta, vpt, numNeighbors) {
    // Compute a single row Pi of the kernel and the Shannon entropy H
    var neighbors = vpt.search(Xi, numNeighbors)
    for (var j = 0; j < neighbors.length; j++) {
      Pi.set(neighbors[j]['i'], Math.exp(-beta * neighbors[j]['d']))
=======
  var euclidean = function (data) {
    return function (x, y) {
      return Math.sqrt(squaredEuclidean(data[x], data[y]))
>>>>>>> nn-cut-2
    }
  }

  var symmetrize = function (P) {
    // Symmetrize in place according to:
    //         p_j|i + p_i|j
    // p_ij = ---------------
    //              2n
    var n = P.shape[0]
    for (var i = 0; i < n; i++) {
      for (var s = 0; s < this.numNeighbors; s++) {
        var j = this.NN.get(i, s)

      }
    }
      for (var j = i + 1; j < n; j++) {
        var Pij = (P.get(i, j) + P.get(j, i)) / (2 * n)
        P.set(i, j, Pij)
        P.set(j, i, Pij)
      }
    }
  }

  function sign (x) { return x > 0 ? 1 : x < 0 ? -1 : 0 }

  var TSNE = function (opt) {}

  TSNE.prototype = {
    profileRecord: {},
    profileStart: function (name) {
      if (!this.profileRecord.hasOwnProperty(name)) {
        this.profileRecord[name] = {
          count: 0,
          time: 0
        }
      }
      this.profileRecord[name].tic = performance.now()
    },
    profileEnd: function (name) {
      var toc = performance.now()
      var record = this.profileRecord[name]
      var elapsed = Math.round(toc - record.tic)
      record.count++
      record.time += (elapsed - record.time) / record.count
      console.log(name + ': ' + elapsed + 'ms' + ', avg ' + Math.round(record.time) + 'ms')
    },
    updateY: function () {
      // Perform gradient update in place
      var alpha = this.iter < 250 ? 0.5 : 0.8
      var n = this.Y.shape[0]
      var dims = this.Y.shape[1]
      var Ymean = [0, 0]  // FIXME: only two dimensional
      for (var i = 0; i < n; i++) {
        for (var d = 0; d < dims; d++) {
          var gradid = this.grad.get(i, d)
          var stepid = this.ytMinus1.get(i, d) - this.ytMinus2.get(i, d)
          var gainid = this.Ygains.get(i, d)

          // Update gain
          var newgain = Math.max(
              sign(gradid) === sign(stepid) ? gainid * 0.8 : gainid + 0.2, 0.01)
          this.Ygains.set(i, d, newgain)

          // Update Y
          var Yid = (this.ytMinus1.get(i, d)
                       - this.learningRate * newgain * gradid
                       + alpha * stepid)
          this.Y.set(i, d, Yid)

          // Accumulate mean for centering
          Ymean[d] += Yid
        }
      }

      // Recenter
      for (var i = 0; i < n; i++) {
        for (var d = 0; d < dims; d++) {
          this.Y.set(i, d, this.Y.get(i, d) - Ymean[d] / n)
        }
      }
    },

    updateQ: function () {
      this.Q = pool.zeros([this.n, this.n])
      this.Qu = pool.zeros([this.n, this.n])
      // Update low dimensional affinities of the embedding
      var n = this.Y.shape[0]
      var dims = this.Y.shape[1]
      var qtotal = 0
      for (var i = 0; i < n; i++) {
        for (var j = i + 1; j < n; j++) {
          var dist = 0
          for (var d = 0; d < dims; d++) {
            var diff = this.Y.get(i, d) - this.Y.get(j, d)
            dist += diff * diff
          }
          var affinity = 1.0 / (1.0 + dist)
          this.Qu.set(i, j, affinity)
          this.Qu.set(j, i, affinity)
          qtotal += 2 * affinity
        }
      }

      // Normalize
      for (var i = 0; i < n; i++) {
        for (var j = i + 1; j < n; j++) {
          var Qij = Math.max(this.Qu.get(i, j) / qtotal, 1e-100)
          this.Q.set(i, j, Qij)
          this.Q.set(j, i, Qij)
        }
      }
      for (var i = 0; i < n; i++) {
        this.Q.set(i, i, 1e-100)
      }
      //debug
      this.qtotal = qtotal
    },

    checkGrad: function () {
      var n = this.Y.shape[0]
      var dims = this.Y.shape[1]
      var epsilon = 1e-5
      for (var i = 0; i < n; i++) {
        for (var d = 0; d < dims; d++) {
          var yold = this.Y.get(i, d)

          this.Y.set(i, d, yold + epsilon)
          this.updateQ()
          var cg0 = computeKL(this.P, this.Q)

          this.Y.set(i, d, yold - epsilon)
          this.updateQ()
          var cg1 = computeKL(this.P, this.Q)

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

    updateGradBH: function () {
      // Early exaggeration
      var exag = this.iter < 250 ? 12 : 1 // todo: this is important... see how can be tuned

      // Initialize quadtree
      var bht = bhtree.BarnesHutTree()
      bht.initWithData(this.Y, this.theta)

      // Compute gradient of the KL divergence
      var n = this.Y.shape[0]
      var dims = this.Y.shape[1]

      // Compute Frep using Barnes-Hut
      var Z = 0
      for (var i = 0; i < n; i++) {
        // NOTE: 2D only
        var Frep = bht.computeForces(this.Y.get(i, 0), this.Y.get(i, 1))
        this.grad.set(i, 0, 4 * Frep.x)
        this.grad.set(i, 1, 4 * Frep.y)
        Z += Frep.Z
      }

      // Compute Fattr over sparse P
      var gradi = new Float64Array(dims)
      for (var i = 0; i < n; i++) {
        // Reset
        for (var d = 0; d < dims; d++) gradi[d] = 0

        // Accumulate Fattr over nearest neighbors
        for (var s = 0; s < this.numNeighbors; s++) {
          var Pij = this.P.get(i, s)
          var Dij = this.D.get(i, s)

          // Unfurled loop, but 2D only
          var mulFactor = 4 * exag * Pij * (1. / (1. + Dij))
          gradi[0] += mulFactor * (this.Y.get(i, 0) - this.Y.get(index, 0))
          gradi[1] += mulFactor * (this.Y.get(i, 1) - this.Y.get(index, 1))
        }
        // Normalize Fattr then increment gradient
        for (var d = 0; d < dims; d++) {
          this.grad.set(i, d, this.grad.get(i, d) / Z + gradi[d])
        }
      }

      return null
    },

    XToNN: function (data) {
      /* Construct a n x numNeighbors matrix, where each element ij is the index of jth nearest neighbor of i in data. Simulataneous a distance matrix D is constructed such that Dij is the distance between X[i] and X[NN[i, j]]. */
      var n = data.length
      this.NN = pool.zeros([n, this.numNeighbors])
      this.D = pool.zeros([n, this.numNeighbors])
      var indices = Array.apply(null, Array(n)).map(function (_, i) { return i })
      var vpt = vptree.build(indices, euclidean(data))
      for (var i = 0; i < n; i++) {
        var neighbors = vpt.search(i, this.numNeighbors + 1) 
        neighbors.shift() // first element is own self
        for (var j = 0; j < neighbors.length; j++) {
          var neighbor = neighbors[j]
          this.NN.set(i, j, neighbor.i)
          this.D.set(i, j, neighbor.d * neighbor.d))
        }
      }
    },

    setPiAndGetH: function (i, beta) {
      // Compute a single row Pi of the kernel and the Shannon entropy H
      var Pi = this.P.pick(i, null)
      for (var j = 0; j < this.numNeighbors; j++) {
        var dist = this.D.get(i, j)
        Pi.set(j, Math.exp(-beta * dist))
      }

      // Normalize
      var sumPi = ops.sum(Pi)
      if (sumPi !== 0) {
        ops.divseq(Pi, sumPi)
      }

      // Compute entropy H
      var H = 0
      for (j = 0; j < this.numNeighbors; j++) {
        var Pji = Pi.get(j)
        // Skip small values to avoid NaNs or exploding values
        if (Pji > 1e-7) { // TODO: do we need this?
          H -= Pji * Math.log2(Pji)
        }
      }

      return H
    },

    DToP: function (perplexity) {
      // Shannon entropy H is log2 of perplexity
      this.P = pool.zeros([this.n, this.numNeighbors])
      var Hdesired = Math.log2(perplexity)

      for (var i = 0; i < this.n; i++) {
        // We perform binary search to find the beta such that
        // the conditional distribution P_i has the given perplexity.
        // We define:
        //   beta = 1 / (2 * sigma_i^2)
        // where sigma_i is the bandwith of the Gaussian kernel
        // for the conditional distribution P_i
        var beta = 1
        var betamin = -Infinity
        var betamax = Infinity
        var Hdiff
        var numTries = 0

        do {
          numTries++
          var H = this.setPiAndGetH(i, beta)
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

      // FIXME: symmetrize
      // Symmetrize conditional distribution
      // symmetrize(P)
    },

    initData: function (data) {
      this.n = data.length
      this.numNeighbors = 3 * 50
      var perplexity = 50  // 30
      this.theta = 0.8  // tunes the barnes-hut approximation, higher is more coarse
      this.XToNN(data)
      this.DToP(perplexity)
      this.Y = initialY(this.n)
      this.ytMinus1 = pool.clone(this.Y)
      this.ytMinus2 = pool.clone(this.Y)
      this.Ygains = pool.ones(this.Y.shape)
      this.iter = 0
      this.learningRate = 200
      this.grad = pool.zeros(this.Y.shape)
    },

    step: function () {
      // Compute gradient
      var cost = this.updateGradBH()
      // if (this.iter > 100) {
      //   this.checkGrad()
      // }

      // Rotate buffers
      var temp = this.ytMinus2
      this.ytMinus2 = this.ytMinus1
      this.ytMinus1 = this.Y
      this.Y = temp

      // Perform update
      this.updateY()

      this.iter++
      return cost
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

