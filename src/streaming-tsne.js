var gaussian = require('gaussian')
var pool = require('ndarray-scratch')
var bhtree = require('./includes/bhtree.js')
var vptree = require('./includes/vptree.js')
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

  var euclidean = function (data) {
    return function (x, y) {
      return Math.sqrt(squaredEuclidean(data[x], data[y]))
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
    updateGradBH: function () {
      // Early exaggeration

      var exag = this.iter < 250 ? 4 : 1 // todo: this is important... see how can be tuned

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
        var that = this
        var pi = this.symP[i]
        for (var k = 0; k < this.numNeighbors; k++) {
          var j = this.NN.get(i, k)
          // Unfurled loop, but 2D only
          var Dij = that.D[i][j] || that.D[j][i]
          var mulFactor = 4 * exag * pi[j] * (1.0 / (1.0 + Dij))
          gradi[0] += mulFactor * (that.Y.get(i, 0) - that.Y.get(j, 0))
          gradi[1] += mulFactor * (that.Y.get(i, 1) - that.Y.get(j, 1))
        }

        // Normalize Fattr then increment gradient
        for (var d = 0; d < dims; d++) {
          this.grad.set(i, d, this.grad.get(i, d) / Z + gradi[d])
        }
      }

      return null
    },

    XToD: function (data) {
      var n = data.length
      var indices = Array.apply(null, Array(n)).map(function (_, i) { return i })
      var vpt = vptree.build(indices, euclidean(data))
      this.NN = pool.zeros([n, this.numNeighbors])
      for (var i = 0; i < n; i++) {
        var neighbors = vpt.search(i, this.numNeighbors + 1)
        neighbors.shift() // first element is own self
        var elem = {}
        for (var j = 0; j < neighbors.length; j++) {
          var neighbor = neighbors[j]
          elem[neighbor.i] = neighbor.d * neighbor.d
          this.NN.set(i, j, neighbor.i)
        }
        this.D.push(elem)
      }
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
      return H
    },

    symmetrizeP: function () {
      // Symmetrize in place according to:
      //         p_j|i + p_i|j
      // p_ij = ---------------
      //              2n
      this.symP = []
      for (var i = 0; i < this.n; i++) {
        this.symP.push({})
      }

      for (var i = 0; i < this.n; i++) {
        var pi = this.P[i]
        for (var k = 0; k < this.numNeighbors; k++) {
          var j = this.NN.get(i, k)
          // if (j === i) { window.alert('not possible') }
          var pji = 0
          if (i in this.P[j]) {
            pji = this.P[j][i]
          }
          var val = (pi[j] + pji) / (2 * this.n)

          // sanity check
          // if (j in this.symP[i]) {
          //  if (val !== this.symP[i][j]) {
          //    window.alert('nooo')
          //  }
          // }
          // if (i in this.symP[j]) {
          //  if (val !== this.symP[j][i]) {
          //    window.alert('nooo')
          //  }
          // }

          this.symP[i][j] = val
          this.symP[j][i] = val
        }
      }
    },

    DToP: function (perplexity) {
      // Shannon entropy H is log2 of perplexity
      var Hdesired = Math.log2(perplexity)
      this.P = []
      for (var i = 0; i < this.n; i++) {
        this.P.push({})
      }

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
      this.symmetrizeP()
    },

    initData: function (data) {
      this.P = []
      this.D = []
      this.n = data.length
      var perplexity = 30  // (van der Maaten 2014)
      this.numNeighbors = 3 * perplexity  // (van der Maaten 2014)
      this.theta = 0.5  // [0, 1] tunes the barnes-hut approximation, 0 is exact
      this.XToD(data)
      this.DToP(perplexity)
      this.Y = initialY(this.n)
      this.ytMinus1 = pool.clone(this.Y)
      this.ytMinus2 = pool.clone(this.Y)
      this.Ygains = pool.ones(this.Y.shape)
      this.iter = 0
      this.learningRate = 10
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

