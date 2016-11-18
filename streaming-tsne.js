const math = require('mathjs')

;(function () {
  'use strict'

  const affinity = function (xI, xJ, sigmaI) {
    const norm = math.norm(math.subtract(xI, xJ))
    return Math.exp(-(norm * norm) / (2 * Math.pow(sigmaI, 2)))
  }

  const conditionalProbabilityMatrix = function (x, sigma) {
    let p = math.zeros([x.length, x.length])
    for (let i = 0; i < x.length; i++) {
      for (let j = 0; j < x.length; j++) {
        if (i === j) continue
        p[i][j] = affinity(x[i], x[j], sigma[i])
      }
      // p[i] = math.divide(p[i], math.sum(p[i]))
    }
    return p
  }

  const jointProbabilityMatrix = function (x, sigma) {
    const pConditional = conditionalProbabilityMatrix(x, sigma)
    let p = math.zeros([x.length, x.length])
    for (let i = 0; i < x.length; i++) {
      for (let j = i + 1; j < x.length; j++) {
        p[i][j] = p[j][i] = (pConditional[i][j] + pConditional[j][i]) / (2 * x.length)
      }
    }
    return p
  }

  const sampleYs = function (numSamples) {
    return math.random([numSamples, 2], -0.0001, 0.0001)
  }

  const lowDimAffinities = function (y) {
    let q = math.zeros([y.length, y.length])
    for (let i = 0; i < y.length; i++) {
      for (let j = i + 1; j < y.length; j++) {
        const norm = math.norm(math.subtract(y[i], y[j]))
        q[i][j] = q[j][i] = 1 / (1 + norm * norm)
      }
    }
    q = math.divide(q, math.sum(q))
    return q
  }

  const gradKL = function (p, q, y) {
    let gradTot = math.zeros(y.length)
    for (let i = 0; i < y.length; i++) {
      let gradI = math.zeros(y.length)
      for (let j = 0; j < y.length; j++) {
        const norm = math.norm(math.subtract(y[i], y[j]))
        gradI[j] = math.multiply((p[i][j] - q[i][j]) / (1 + norm * norm), math.subtract(y[i], y[j]))
      }
      gradTot[i] = 4 * math.sum(gradI)
    }
    return gradTot
  }

  const numSamples = 10
  const dimension = 10
  const x = math.random([numSamples, dimension], -100, 100)
  const sigma = math.multiply(math.ones([numSamples]), 100)
  const p = jointProbabilityMatrix(x, sigma)
  const y = sampleYs(numSamples)
  const q = lowDimAffinities(y)
  let grad = gradKL(p, q, y)
  console.log(grad)
// console.log(conditionalProbability(0, 1, [[1], [1], [1], [1]], [1, 1, 1, 1]))
})()
