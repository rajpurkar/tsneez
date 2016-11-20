const math = require('mathjs')
const gaussian = require('gaussian')
const data = require('./wordvecs50dtop1000.json')

;(function () {
  'use strict'

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
        const diff = math.subtract(y[i], y[j])
        const affinity = 1.0 / (1 + math.dot(diff, diff))
        q[i][j] = q[j][i] = math.max(affinity, 1e-12)
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
        const diff = math.subtract(y[i], y[j])
        gradI[j] = math.multiply((p[i][j] - q[i][j]) / (1 + math.multiply(diff, diff)), diff)
      }
      gradTot[i] = math.multiply(4, math.flatten(math.multiply(math.transpose(gradI), math.ones([y.length, 1]))))
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
        cost += p[i][j] * math.log(p[i][j] / q[i][j])
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

    var getpIAndH = function (dI, betaI) {
      let pI = math.exp(math.multiply(-betaI, dI))
      const sumP = math.sum(pI)
      const H = math.log(sumP) + betaI * math.multiply(dI, pI) / sumP
      pI = math.divide(pI, sumP)
      return [pI, H]
    }

    for (let i = 0; i < n; i++) {
      let betamin = -math.Infinity
      let betamax = math.Infinity
      const dI = D[i]
      let Hdiff, pI
      do {
        const pIAndH = getpIAndH(dI, beta[i])
        pI = pIAndH[0]
        let H = pIAndH[1]
        Hdiff = H - logU

        if (Hdiff > 0) {
          betamin = beta[i]
          if (betamax === math.Infinity) {
            beta[i] = beta[i] * 2
          } else {
            beta[i] = (beta[i] + betamax) / 2
          }
        } else {
          betamax = beta[i]
          if (betamin === -math.Infinity) {
            beta[i] = beta[i] / 2
          } else {
            beta[i] = (beta[i] + betamin) / 2
          }
        }
      } while (math.abs(Hdiff) > 1e-05)
      P[i] = pI
    }
    console.log('Sigma mean: ' + math.mean(math.sqrt(math.dotDivide(1, beta))))
    return P
  }

  const symmetrizeP = function (P) {
    P = math.add(P, math.transpose(P))
    P = math.multiply(math.divide(P, math.sum(P)), 4)
    return P
  }

  const numIterations = 10
  const x = data.vecs.splice(0, 100)
  const numSamples = x.length
  const D = xToD(x)
  const pUnsymmetrized = DToP(D, 40)
  const p = symmetrizeP(pUnsymmetrized)
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
