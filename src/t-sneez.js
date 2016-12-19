var gaussian = require('gaussian')
var pool = require('ndarray-scratch')
var ops = require('ndarray-ops')
var bhtree = require('./includes/bhtree.js')
var vptree = require('./includes/vptree.js')
var tsneez = tsneez || {}

;(function (global) {
  'use strict'

  var getopt = function (opt, key, def) {
    if (opt[key] == null) {
      return def
    } else {
      return opt[key]
    }
  }

  var squaredEuclidean = function (x, y) {
    // Compute Euclidean distance between two vectors as Arrays
    var m = x.length
    var d = 0
    var k, xk, yk
    for (k = 0; k < m; k++) {
      xk = x[k]
      yk = y[k]
      d += (xk - yk) * (xk - yk)
    }
    return d
  }

  var euclideanOf = function (data) {
    return function (x, y) {
      return Math.sqrt(squaredEuclidean(data[x], data[y]))
    }
  }

  function sign (x) { return x > 0 ? 1 : x < 0 ? -1 : 0 }

  var TSNEEZ = function (opt) {
    var perplexity = getopt(opt, 'perplexity', 30)  // (van der Maaten 2014)
    this.Hdesired = Math.log2(perplexity)
    this.numNeighbors = getopt(opt, 'numNeighbors', 3 * perplexity)  // (van der Maaten 2014)
    this.theta = getopt(opt, 'theta', 0.5)  // [0, 1] tunes the barnes-hut approximation, 0 is exact
    this.learningRate = getopt(opt, 'learningRate', 10)  // [0, 1] tunes the barnes-hut approximation, 0 is exact
  }

  TSNEEZ.prototype = {
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
    initY: function () {
      // FIXME: allow arbitrary dimensions??
      var distribution = gaussian(0, 1e-4)
      var ys = pool.zeros([this.n * 2, 2])  // initialize with twice as much room as necessary
      for (var i = 0; i < this.n; i++) {
        ys.set(i, 0, distribution.ppf(Math.random()))
        ys.set(i, 1, distribution.ppf(Math.random()))
      }
      return ys
    },
    updateY: function () {
      // Perform gradient update in place
      // var alpha = this.iter < this.exagEndIter ? 0.5 : 0.8
      // var alpha = 0.7
      var alpha = 0.9 // TODO look at different learning rates / annealing
      var n = this.n
      var dims = this.dims
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
    updateGradBH: function () {
      // Early exaggeration
      // var exag = Math.max(8 - 0.4 * Math.sqrt(this.iter), 0.2) // spent lot of time tuning this
      var exag = Math.max(8 - 0.4 * Math.sqrt(this.iter), 1) // spent lot of time tuning this
      // var exag = this.iter < this.exagEndIter ? 12 - (0.01 * this.iter) : 1 // todo: this is important... see how can be tuned

      // Initialize quadtree
      var bht = bhtree.BarnesHutTree()
      bht.initWithData(this.Y, this.theta, this.n)

      // Compute gradient of the KL divergence
      var n = this.n
      var dims = this.dims

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
        var pi = this.P[i]
        for (var k = 0; k < this.numNeighbors; k++) {
          var j = this.NN.get(i, k)
          var Dij = this.D[i][j]

          // Symmetrize on-demand
          var Pij = (pi[j] + (this.P[j][i] || 0)) / (2 * this.n)

          // Unfurled loop, but 2D only
          var mulFactor = 4 * exag * Pij * (1.0 / (1.0 + Dij))
          gradi[0] += mulFactor * (this.Y.get(i, 0) - this.Y.get(j, 0))
          gradi[1] += mulFactor * (this.Y.get(i, 1) - this.Y.get(j, 1))
        }

        // Normalize Frep then increment gradient
        for (var d = 0; d < dims; d++) {
          this.grad.set(i, d, this.grad.get(i, d) / Z + gradi[d])
        }
      }

      return null
    },

    XToD: function () {
      var indices = Array.apply(null, Array(this.n)).map(function (_, i) { return i })
      this.vpt = vptree.build(indices, euclideanOf(this.X))
      this.D = []
      this.dmax = []
      this.kmax = []
      for (var i = 0; i < this.n; i++) {
        this.pushD(i)
      }
    },

    pushD: function(i) {
      var neighbors = this.vpt.search(i, this.numNeighbors + 1)
      neighbors.shift() // first element is own self
      var elem = {}
      var dmaxi = 0
      var kmaxi
      for (var j = 0; j < neighbors.length; j++) {
        var neighbor = neighbors[j]
        elem[neighbor.i] = neighbor.d * neighbor.d
        this.NN.set(i, j, neighbor.i)

        // Keep track of maximum distance
        if (neighbor.d > dmaxi) {
          dmaxi = neighbor.d
          kmaxi = neighbor.i
        }
      }
      this.D.push(elem)
      this.dmax.push(dmaxi)
      this.kmax.push(kmaxi)
    },

    setPiAndGetH: function (i, beta) {
      // Compute a single row Pi of the kernel and the Shannon entropy H

      var pi = {}
      var sum = 0
      var Di = this.D[i]
      for (var k = 0; k < this.numNeighbors; k++) {
        var key = this.NN.get(i, k)
        var elem = Math.exp(-beta * Di[key])
        pi[key] = elem
        sum += elem
      }

      // For debugging
      // if (sum === 0) {
      //  console.count('sum equals zero')
      // }

      var H = 0
      for (var k = 0; k < this.numNeighbors; k++) {
        var key = this.NN.get(i, k)
        var val = pi[key] / sum
        pi[key] = val
        if (val > 1e-7) { // TODO: do we need this?
          H -= val * Math.log2(val)
        }
      }

      this.P[i] = pi
      this.Psum[i] = sum
      return H
    },

    DToP: function () {
      this.P = []
      this.Psum = []
      this.beta = []

      for (var i = 0; i < this.n; i++) {
        this.pushP(i)
      }
    },

    pushP: function (i) {
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
        Hdiff = H - this.Hdesired

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
      this.beta.push(beta)
    },

    /* Update neighborhoods of other points
     *
     * newj - index of the new point
     */
    updateNeighborhoods: function (newj) {
      var i, newd, kmax, dmax, jmax, newdSq
      for (i = 0; i < newj; i++) {
        dmax = this.dmax[i]
        newdSq = squaredEuclidean(this.X[newj], this.X[i])
        newd = Math.sqrt(newdSq)

        if (newd < dmax) {
          // Xnewj is in the neighborhood of Xi!
          // Replace the point farthest away from Xi in neighborhood
          kmax = this.kmax[i]
          jmax = this.NN[kmax]
          this.NN[kmax] = newj
          delete this.P[i][jmax]  // or this.P[i][jmax] = 0?

          // Compute approximate update with old beta and Psum
          // (Note that to compute an exact update, we have to redo the
          // search for beta, or at least renormalize Pi)
          this.P[i][newj] = Math.exp(-this.beta[i] * newdSq) / this.Psum[i]
        }
      }
    },

    pushY: function(newi) {
      // Initialize embedding as weighted average of its neighbors (Pezzotti)
      var Pi = this.P[newi]
      var y0 = 0
      var y1 = 0
      var k, j, Pji
      for (k = 0; k < this.numNeighbors; k++) {
        j = this.NN.get(newi, k)
        Pji = Pi[j]
        y0 = Pji * this.Y.get(j, 0)
        y1 = Pji * this.Y.get(j, 1)
      }
      this.Y.set(newi, 0, y0)
      this.Y.set(newi, 1, y1)
    },

    expandBuffers: function() {
      console.log('expanding buffers')
      var newlen = this.n * 2
      var that = this
      ;['NN', 'Y', 'ytMinus1', 'ytMinus2', 'Ygains', 'grad'].forEach(function (name) {
        var oldMat = that[name]
        var newMat = pool.malloc([newlen, oldMat.shape[1]])
        ops.assign(newMat.hi(oldMat.shape[0], oldMat.shape[1]), oldMat)
        pool.free(oldMat)
        that[name] = newMat
      })
    },

    /************************
     * PUBLIC API STARTS HERE
     ************************/

    initData: function (data) {
      this.X = data
      this.n = data.length
      this.NN = pool.zeros([this.n * 2, this.numNeighbors])
      this.dims = 2
      this.XToD()
      this.DToP()
      this.Y = this.initY()
      this.ytMinus1 = pool.clone(this.Y)
      this.ytMinus2 = pool.clone(this.Y)
      this.Ygains = pool.ones(this.Y.shape)
      this.grad = pool.zeros(this.Y.shape)
      this.iter = 0
      this.exagEndIter = 250  // van der Maaten 2014
    },

    /*
     * x - array containing the new point
     */
    add: function (x) {
      if (x.length !== this.X[0].length) {
        console.log("New point doesn't match input dimensions")
        return
      }
      var newi = this.n++
      this.X.push(x)
      if (this.n > this.Y.shape[0]) {
        // Expand buffers and rebuild P
        this.expandBuffers()
        this.XToD()
        this.DToP()
        // this.exagEndIter = this.iter + 100  // exaggerate for another 100 iterations
      } else {
        // Do an approximative update
        this.updateNeighborhoods(newi)
        this.pushD(newi)
        this.pushP(newi)
        this.pushY(newi)
      }
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

  global.TSNEEZ = TSNEEZ
})(tsneez)

// export the library to window, or to module in nodejs
// Webpack supports both.
;(function (lib) {
  'use strict'
  if (typeof module !== 'undefined' && typeof module.exports === 'undefined') {
    module.exports = lib // in nodejs
  }
  if (typeof window !== 'undefined') {
    window.tsneez = lib // in ordinary browser attach library to window
  }
})(tsneez)

