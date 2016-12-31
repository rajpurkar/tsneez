var gaussian = require('gaussian')
var pool = require('ndarray-scratch')
var bhtree = require('./includes/bhtree.js')
var vptree = require('./includes/vptree.js')

var TSNEEZ = function (opt) {
  this.perplexity = getopt(opt, 'perplexity', 10)  // (van der Maaten 2014)
  this.numNeighbors = getopt(opt, 'numNeighbors', 3 * this.perplexity)  // (van der Maaten 2014)
  this.theta = getopt(opt, 'theta', 0.5)  // [0, 1] tunes the barnes-hut approximation, 0 is exact
  this.learningRateStart = getopt(opt, 'learningRateStart', 5)
  this.earlyExaggerationStart = getopt(opt, 'earlyExaggerationStart', 8)
  this.dims = getopt(opt, 'dims', 2)
}

var getopt = function (opt, key, def) {
  if (opt === undefined || opt[key] === undefined) {
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

TSNEEZ.prototype = {
  initYGaussian: function (start) {
    var distribution = gaussian(0, 1e-4)
    for (var i = start; i < this.n; i++) {
      for (var j = 0; j < this.dims; j++) {
        this.Y.set(i, j, distribution.ppf(Math.random()))
        this.Y.set(i, j, distribution.ppf(Math.random()))
      }
    }
  },
  updateY: function () {
    // Perform gradient update in place
    var momentum = 0.9
    var n = this.n
    var dims = this.dims
    var Ymean = pool.zeros([this.dims])
    var lr = Math.max(this.learningRateStart * Math.pow(0.99, this.iter), 0.01)
    for (var i = 0; i < n; i++) {
      for (var d = 0; d < dims; d++) {
        var gradid = Math.max(Math.min(this.grad.get(i, d), 5), -5) // clip the gradients and 5 and -5. TODO: experimentally tune these clips
        var stepid = this.ytMinus1.get(i, d) - this.ytMinus2.get(i, d)
        var gainid = this.Ygains.get(i, d)

        // Update gain

        var newgain = Math.max(
            sign(gradid) === sign(stepid) ? gainid * 0.8 : gainid + 0.2, 0.01)
        this.Ygains.set(i, d, newgain)
        // Update Y
        var Yid = (this.ytMinus1.get(i, d)
                     - lr * newgain * gradid
                     + momentum * stepid)
        this.Y.set(i, d, Yid)

        // Accumulate mean for centering
        Ymean.set(d, Ymean.get(d) + Yid)
      }
    }

    // Recenter
    for (var i = 0; i < n; i++) {
      for (var d = 0; d < dims; d++) {
        this.Y.set(i, d, this.Y.get(i, d) - Ymean.get(d) / n)
      }
    }
  },
  updateGradBH: function () {
    // Early exaggeration
    var exag = Math.max(this.earlyExaggerationStart * Math.pow(0.99, this.iter), 1)

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

  pushD: function (i) {
    var neighbors = this.vpt.search(i, this.numNeighbors + 1)
    if (neighbors.length === 0) {
      console.error('no neighbors found for element ' + i)
    }
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
      Hdiff = H - Math.log2(this.perplexity)

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

  pushY: function (newi) {
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

  /************************
   * PUBLIC API STARTS HERE
   ************************/

  initData: function (X, reinit, oldN) {
    this.X = X
    this.n = this.X.length
    this.largeDims = this.X[0].length
    if (this.n - 1 < this.numNeighbors) {
      this.numNeighbors = this.n - 1
      this.perplexity = Math.max(this.n - 2, 1)
    }

    if (this.NN) pool.free(this.NN)
    this.NN = pool.zeros([this.n + 100, this.numNeighbors]) // init w 100 extra spots

    var Y = pool.zeros([this.n + 100, this.dims])  // init w 100 extra spots
    var Ygains = pool.ones(Y.shape)
    var grad = pool.zeros(Y.shape)
    var that = this
    if (reinit) {
      ['Y', 'Ygains', 'grad'].forEach(function (name) {
        var newMat = eval(name)
        var oldMat = that[name]
        for (var i = 0; i < oldMat.shape[0]; i++) {
          for (var j = 0; j < oldMat.shape[1]; j++) {
            newMat.set(i, j, oldMat.get(i, j))
          }
        }
        pool.free(oldMat)
      })
    }
    this.Y = Y
    this.Ygains = Ygains
    this.grad = grad

    if (reinit) {
      this.initYGaussian(oldN + 1)
      this.iter = 0
    } else {
      this.initYGaussian(0)
      this.iter = 0
    }

    if (reinit) {
      pool.free(this.ytMinus1)
      pool.free(this.ytMinus2)
    }
    this.ytMinus1 = pool.clone(this.Y)
    this.ytMinus2 = pool.clone(this.Y)

    this.XToD()
    this.DToP()
  },

  /*
   * XNew - array containing the new points
   */
  addPoints: function (XNew) {
    if (XNew[0].length !== this.X[0].length) {
      console.log("New points don't match input dimensions")
      return
    }

    var newLength = this.n + XNew.length

    if (newLength > this.Y.shape[0]) {
      this.X.push.apply(this.X, XNew)
      this.initData(this.X, true, XNew.length)
    } else {
      // FIXME: bump up learning rate
      this.X.push.apply(this.X, XNew)
      // Do approximative updates for each point
      for (var i = this.n; i < newLength; i++) {
        this.n++
        this.updateNeighborhoods(i)
        this.pushD(i)
        this.pushP(i)
        this.pushY(i)
      }
    }
  },

  step: function () {
    // Compute gradient
    this.updateGradBH()

    // Rotate buffers
    var temp = this.ytMinus2
    this.ytMinus2 = this.ytMinus1
    this.ytMinus1 = this.Y
    this.Y = temp

    // Perform update
    this.updateY()

    this.iter++
  }
}

module.exports = {TSNEEZ: TSNEEZ}
