(function (tsneez, $, d3, performance, karpathy_tsne, scienceai_tsne, randomColor) {
  var data
  var DO_PROFILE = false
  var DO_TIME = false
  var METHOD = 'tsneez'
  var buildDir = '/tsneez'
  var scienceaiWorkerPath = buildDir + '/javascripts/scienceai-worker.js'
  var DATA_PATH = buildDir + '/data/shortglove.json'
  var N = 700
  var stepnum = 0
  var PERPLEXITY = 10
  

  // Multiplex between methods
  var T, getEmbedding, initData, stepEmbedding
  switch (METHOD) {
    case 'tsneez':
      T = new tsneez.TSNEEZ({
        theta: 0.5,
        perplexity: PERPLEXITY,
        randomProjectionInitialize: false
      })
      initData = function (vecs) { T.initData(vecs) }
      stepEmbedding = function () { stepnum++; return T.step() }
      getEmbedding = function () { return T.Y }
      break
    case 'karpathy':
      T = new karpathy_tsne.tSNE({
        perplexity: PERPLEXITY
      })
      initData = function (vecs) { T.initDataRaw(vecs) }
      stepEmbedding = function () { stepnum++; return T.step() }
      getEmbedding = function () {
        // Return wrapper around the nested arrays to match ndarray API
        var Y = T.getSolution()
        return {
          get: function (i, d) {
            return Y[i][d]
          }
        }
      }
      break
    case 'scienceai':
      var Tworker = new Worker(scienceaiWorkerPath)
      var Ycurrent = null
      var tic = performance.now()
      Tworker.onmessage = function (e) {
        var msg = e.data
        switch (msg.type) {
          case 'PROGRESS_STATUS':
            break
          case 'PROGRESS_ITER':
            break
          case 'PROGRESS_DATA':
            // Do our own custom profiling
            var toc = performance.now()
            if (DO_TIME === true) {
              if (Ycurrent === null) {
                console.log('initialization', (toc - tic) + 'ms')
              } else {
                console.log('step', (toc - tic) + 'ms')
              }
            }
            Ycurrent = msg.data
            tic = performance.now()
            stepnum++
            break
          case 'STATUS':
            console.log('status', msg.data)
            break
          case 'DONE':
            Ycurrent = msg.data
            break
          default:
        }
      }

      initData = function (vecs) {
        Tworker.postMessage({
          type: 'INPUT_DATA',
          data: vecs
        })
        Tworker.postMessage({
          type: 'RUN',
          data: {
            perplexity: 30,
            earlyExaggeration: 4,
            learningRate: 10,
            nIter: 1000,
            metric: 'euclidean'
          }
        })
      }
      stepEmbedding = function () { return }
      getEmbedding = function () { 
        if (Ycurrent === null) {
          return { get: function () { return null } }
        } else {
          return {
            get: function (i, d) {
              return Ycurrent[i][d]
            }
          }
        }
      }
      // Turn off normal profiling
      DO_TIME = false
      break
  }

  // Update d3 embedding on a step
  function updateEmbedding () {
    var Y = getEmbedding()
    if (Y === null) return  // scienceai might not be ready
    var s = svg.selectAll('.u')
    .data(data.words)
    .attr('transform', function (d, i) {
      if (!d.rotate) {
        d.rotate = (Math.random() - 0.5) * 10
      } else {
        d.rotate = d.rotate
      }
      return 'translate(' +
        ((Y.get(i, 0) * 200 * ss + tx) + 400) + ',' +
        ((Y.get(i, 1) * 200 * ss + ty) + 400) + ')' +
        'rotate(' + d.rotate + ')'
    })

    s.selectAll('rect').style('fill-opacity', function (d) {
      if (d.init === true && fadeOld > 0) {
        return Math.max(Math.min(0.9 - Math.sqrt(fadeOld * 0.09), 0.9), 0)
      } else {
        return 0.9
      }
    })
    fadeOld--
  }

  // Resize the viewport in response to window resize
  function resize () {
    var width = $('.viewport').width()
    var height = 600
    svg.attr('width', width).attr('height', height)
  }

  // Set up visualization
  var svg
  var fadeOld = 0
  var zoomListener = d3.behavior.zoom()
  .scaleExtent([0.0005, 10])
  .center([0, 0])
  .on('zoom', zoomHandler)
  var tx = 0
  var ty = 0
  var ss = 1
  function zoomHandler () {
    tx = d3.event.translate[0]
    ty = d3.event.translate[1]
    ss = d3.event.scale
  }

  // (re-)Draw the visualization
  function draw () {
    var g = svg.selectAll('.b')
      .data(data.words)//, function (d) { return d.str })
      .enter().append('g')
      .attr('class', 'u')
    g.append('rect')
    .attr('width', function (d) { return (d.str.length * 8) + 10 })
    .attr('height', 20)
    .attr('rx', 5)
    .attr('ry', 5)
    .style('fill', function (d) {
      return randomColor({luminosity: 'light', seed: d.str})
    })

    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('x', function (d) { return (d.str.length * 4) + 5 })
      .attr('y', 10)
      .attr('alignment-baseline', 'central')
      .attr('fill', '#333')
      .text(function (d) { return d.str })
  }

  // Draw initial embedding
  function drawEmbedding () {
    var div = d3.select('.viewport')
    svg = div.append('svg') // svg is global
    draw()
    zoomListener(svg)
    d3.select(window).on('resize', resize)
    resize()
  }

  // Step the embedding and visualization
  var tic = performance.now()
  function step () {
    DO_TIME && console.time('step')
    stepEmbedding()
    DO_TIME && console.timeEnd('step')
    var fps = Math.round((stepnum / (performance.now() - tic)) * 1000)
    updateEmbedding()

    if (stepnum === 10) {
      if (DO_PROFILE && window.console && window.console.profile) {
        console.profileEnd()
      }
    }

    window.requestAnimationFrame(step)
  }

  $(window).load(function () {
    $.getJSON(DATA_PATH, function (j) {
      // Wrap words in objects with metadata
      j.words = j.words.map(function (word) {
        return {str: String(word), init: true}
      })

      data = {
        words: j.words.slice(0, N),
        vecs: j.vecs.slice(0, N)
      }

      // Initialize the t-SNE model
      if (DO_PROFILE && window.console && window.console.profile) {
        console.profile('initialization')
      }
      DO_TIME && console.time('initialization')
      initData(data.vecs)
      DO_TIME && console.timeEnd('initialization')
      if (DO_PROFILE && window.console && window.console.profile) {
        console.profileEnd()
      }

      // Initialize the visualization
      drawEmbedding()

      // Start the animation
      if (DO_PROFILE && window.console && window.console.profile) {
        console.profile('step')
      }
      window.requestAnimationFrame(step)

      // Set up listener for adding points
      $('#addPoints').click(function () {
        window.cancelAnimationFrame(step)
        fadeOld = 10
        data.words = data.words.map(function (word) {
          word.init = true
          return word
        })

        var numAdd = 10
        T.addPoints(j.vecs.slice(N, N + numAdd + 1))

        for (var i = 0; i < numAdd; i++) {
          var word = j.words[N]
          word.init = false
          data.words.push(word)
          N++
        }

        d3.selectAll('.viewport > svg').remove()
        drawEmbedding()
        window.requestAnimationFrame(step)
      })
    }).fail(function (d, textStatus, error) {
      console.log('getJSON failed, status: ' + textStatus + ', error: ' + error)
    })
  })
})(tsneez, $, d3, performance, tsnejs, TSNE, randomColor)
