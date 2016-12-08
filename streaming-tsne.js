const math = require('mathjs')
const gaussian = require('gaussian')
const data = require('./wordvecs50dtop1000.json')

var tsne = tsne || {}

;(function (global) {
  'use strict'

  var seed = 1
  const tmpRandom = function () {
    const x = Math.sin(seed++) * 10000
    return x - Math.floor(x)
  }

  const sampleYs = function (numSamples) {
    const distribution = gaussian(0, 1e-4)
    const ys = math.zeros([numSamples, 2])
    for (let i = 0; i < numSamples; i++) {
      ys[i] = [distribution.ppf(tmpRandom()), distribution.ppf(tmpRandom())]
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

  // Get Pairwise distances. Based on whole squared expansion (a - b)^2 = a^2 - 2ab + b^2, where a and b are rows of x.
  const xToD = function (x) {
    const xSquared = math.square(x)
    const onesXX = math.ones([x[0].length, x.length])
    const xSquaredMat = math.multiply(xSquared, onesXX)
    const xIxJPairTerm = math.multiply(-2, math.multiply(x, math.transpose(x)))
    const D = math.add(math.add(xIxJPairTerm, xSquaredMat), math.transpose(xSquaredMat))
    return D
  }

  const DToP = function (D, perplexity) {
    const n = math.size(D)[0]
    const P = math.zeros([n, n])
    const beta = math.ones([n])
    const logU = math.log(perplexity)

    var getpIAndH = function (dI, betaI, i) {
      let pI = math.exp(math.multiply(-betaI, dI))
      pI[i] = 0
      const sumP = math.sum(pI)
      let H = 0.0
      for (var j = 0; j < math.size(dI)[0]; j++) {
        if (sumP === 0) {
          pI[j] = 0
        } else {
          pI[j] = pI[j] / sumP
        }
        if (pI[j] > 1e-7) H -= pI[j] * Math.log(pI[j])
      }
      // const H = math.log(sumP) + betaI * math.multiply(dI, pI) / sumP
      // pI = math.divide(pI, sumP)
      return [pI, H]
    }

    for (let i = 0; i < n; i++) {
      let betamin = -Infinity
      let betamax = Infinity
      const dI = D[i]
      let Hdiff, pI
      let numTries = 0
      do {
        numTries++
        const pIAndH = getpIAndH(dI, beta[i], i)
        pI = pIAndH[0]
        let H = pIAndH[1]
        Hdiff = H - logU

        if (Hdiff > 0) {
          betamin = beta[i]
          if (betamax === Infinity) {
            beta[i] = beta[i] * 2
          } else {
            beta[i] = (beta[i] + betamax) / 2
          }
        } else {
          betamax = beta[i]
          if (betamin === -Infinity) {
            beta[i] = beta[i] / 2
          } else {
            beta[i] = (beta[i] + betamin) / 2
          }
        }
      } while (math.abs(Hdiff) > 1e-05 && numTries < 50)
      P[i] = pI
    }
    console.log('Sigma mean: ' + math.mean(math.sqrt(math.dotDivide(1, beta))))
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

  const symmetrizeP = function (P) {
    P = math.add(P, math.transpose(P))
    P = math.divide(P, math.sum(P))
    return P
  }

  let tsne = function (opt) {}

  tsne.prototype = {
    initData: function (data) {
      const numSamples = data.length
      const D = xToD(data)
      const pUnsymmetrized = DToP(D, 30)
      this.p = symmetrizeP(pUnsymmetrized)
      this.y = sampleYs(numSamples)
      this.ytMinus1 = this.y
      this.ytMinus2 = this.y
    },
    step: function () {
      const q = lowDimAffinities(this.y)
      const cost = computeCost(this.p, q)
      console.log('Cost: ' + cost)
      let grad = gradKL(this.p, q, this.y)
      debugGrad(grad, cost, this.y, this.p)
      this.yTMinus2 = this.ytMinus1
      this.ytMinus1 = this.y
      this.y = updateY(grad, this.ytMinus1, this.yTMinus2)
    }
  }

  global.tsne = tsne
})(tsne)

// export the library to window, or to module in nodejs
;(function (lib) {
  'use strict'
  if (typeof module === 'undefined' || typeof module.exports === 'undefined') {
    window.tsne = lib // in ordinary browser attach library to window
  } else {
    module.exports = lib // in nodejs
  }
})(tsne)

var T = new tsne.tsne()
T.initData(data.vecs.splice(0, 100))
for (let i = 0; i < 10; i++) {
  T.step()
}
console.log(T.y)
