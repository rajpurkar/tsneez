const math = require('mathjs')
const gaussian = require('gaussian')
const data = require('./wordvecs50dtop1000.json')

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
      p[i] = math.divide(p[i], math.sum(p[i]))
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
    const distribution = gaussian(0, 0.0001)
    const ys = math.zeros([numSamples, 2])
    for (let i = 0; i < numSamples; i++) {
      ys[i] = [distribution.ppf(Math.random()), distribution.ppf(Math.random())]
    }
    return ys
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
    let gradTot = math.zeros([y.length])
    for (let i = 0; i < y.length; i++) {
      let sum = 0
      for (let j = 0; j < y.length; j++) {
        const norm = math.norm(math.subtract(y[i], y[j]))
        sum += math.multiply((p[i][j] - q[i][j]) / (1 + norm * norm), math.subtract(y[i], y[j]))[0]
      }
      gradTot[i] = 4.0 * sum
    }
    return gradTot
  }

  const x = data.vecs.splice(0, 20)
  const numSamples = x.length
  const sigma = math.multiply(math.ones([numSamples]), 1)
  const p = jointProbabilityMatrix(x, sigma)
  // console.log(p)
  const y = sampleYs(numSamples)
  const q = lowDimAffinities(y)
  // console.log(q)
  let grad = gradKL(p, q, y)
  console.log(grad)
  // console.log(conditionalProbability(0, 1, [[1], [1], [1], [1]], [1, 1, 1, 1]))
})()
