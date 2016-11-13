const math = require('mathjs')

;(function () {
  'use strict'

  const affinity = function (xI, xJ, sigmaI) {
    const norm = math.norm(xI - xJ)
    return Math.exp(-(norm * norm) / (2 * Math.pow(sigmaI, 2)))
  }

  const conditionalProbability = function (i, j, x, sigma) {
    // TODO: optimize this
    let denom = 0
    for (let k = 0; k < x.length; k++) {
      if (k === i) continue
      denom += affinity(x[i], x[k], sigma[i])
    }
    return affinity(x[i], x[j], sigma[i]) / denom
  }

  const conditionalProbabilityMatrix = function (x, sigma) {
    let p = math.zeros(x.length, x.length)
    for (let i = 0; i < x.length; i++) {
      for (let j = 0; j < x.length; j++) {
        if (i === j) continue
        p[i][j] = conditionalProbability(i, j, x, sigma)
      }
    }
  }

  const jointProbabilityMatrix = function (x, sigma) {
    const pConditional = conditionalProbabilityMatrix(x, sigma)
    let p = math.zeros(x.length, x.length)
    for (let i = 0; i < x.length; i++) {
      for (let j = i + 1; j < x.length; j++) {
        p[i][j] = p[j][i] = (pConditional[i][j] + pConditional[j][i]) / (2 * x.length)
      }
    }
  }

  console.log(conditionalProbability(0, 1, [[1], [1], [1], [1]], [1, 1, 1, 1]))
})()
