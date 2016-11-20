const math = require('mathjs')
const gaussian = require('gaussian')
const data = require('./wordvecs50dtop1000.json')

;(function () {
  'use strict'

  const computeAffinity = function (xI, xJ, sigmaI) {
    const norm = math.norm(math.subtract(xI, xJ))
    return Math.exp(-(norm * norm) / (2 * Math.pow(sigmaI, 2)))
  }

  const conditionalProbabilityMatrix = function (x, sigma) {
    let p = math.zeros([x.length, x.length])
    for (let i = 0; i < x.length; i++) {
      for (let j = 0; j < x.length; j++) {
        if (i === j) continue
        const affinity = computeAffinity(x[i], x[j], sigma[i])
        p[i][j] = Math.max(affinity, 1e-12)
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
        const affinity = 1 / (1 + norm * norm)
        q[i][j] = q[j][i] = Math.max(affinity, 1e-12)
      }
    }
    q = math.divide(q, math.sum(q))
    return q
  }

  const gradKL = function (p, q, y) {
    let gradTot = math.zeros([y.length, 2])
    for (let i = 0; i < y.length; i++) {
      let gradI = math.zeros([y.length, 2])
      for (let j = 0; j < y.length; j++) {
        const norm = math.norm(math.subtract(y[i], y[j]))
        gradI[j] = math.multiply((p[i][j] - q[i][j]) / (1 + norm * norm), math.subtract(y[i], y[j]))
      }
      for (let column = 0; column < math.size(gradI)[1]; column++) {
        let gradIcol = math.subset(gradI, math.index(math.range(0, math.size(gradI)[0]), column))
        gradTot[i][column] = 4 * math.sum(gradIcol)
      }
    }
    return gradTot
  }

  const updateY = function (grad, ytMinus1, ytMinus2) {
    const learningRate = 100
    const alpha = 0.5
    const yT = math.add(math.add(ytMinus1, math.multiply(learningRate, grad)), math.multiply(alpha, math.subtract(ytMinus1, yTMinus2)))
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

  const numIterations = 10
  const x = data.vecs.splice(0, 100)
  const numSamples = x.length
  const sigma = math.multiply(math.ones([numSamples]), 0.2)
  const p = jointProbabilityMatrix(x, sigma)
  let y = sampleYs(numSamples)
  var ytMinus1, yTMinus2
  ytMinus1 = yTMinus2 = y
  for (let iteration = 0; iteration < numIterations; iteration++) {
    const q = lowDimAffinities(y)
    const cost = computeCost(p, q)
    let grad = gradKL(p, q, y)
    yTMinus2 = ytMinus1
    ytMinus1 = y
    y = updateY(grad, ytMinus1, yTMinus2)
    console.log(cost)
  }
  console.log(y)
})()
