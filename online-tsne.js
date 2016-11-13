const math = require('mathjs')

;(function () {
  'use strict'

  var affinity = function(x_i, x_j, sigma_i) {
    const norm = math.norm(x_i - x_j)
    return Math.exp(-(norm * norm) / (2 * Math.pow(sigma_i, 2)))
  }

  var conditional_probability = function (i, j, x, sigma) {
    // TODO: optimize this
    var denom = 0
    for (let k = 0; k < x.length; k++) {
      if (k === i) continue
      denom += affinity(x[i], x[k], sigma[i])
    }
    return affinity(x[i], x[j], sigma[i]) / denom
  }


  console.log(conditional_probability(0, 1, [[1], [1], [1], [1]], [1, 1, 1, 1]))

})()
