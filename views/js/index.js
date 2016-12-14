(function (tsne, $, d3, performance, tsnejs) {
  var T = new tsne.TSNE({
    theta: 0.8
  }) // create a tSNE instance
  var data

  function updateEmbedding () {
    var Y = T.Y
    svg.selectAll('.u')
    .data(data.words)
    .attr('transform', function (d, i) {
      return 'translate(' +
        ((Y.get(i, 0) * 200 * ss + tx) + 400) + ',' +
        ((Y.get(i, 1) * 200 * ss + ty) + 400) + ')' })
  }

  function resize () {
    var width = $('.viewport').width()
    var height = 400
    svg.attr('width', width).attr('height', height)
  }

  var svg
  function drawEmbedding () {
    var div = d3.select('.viewport')

    svg = div.append('svg') // svg is global

    var g = svg.selectAll('.b')
      .data(data.words)
      .enter().append('g')
      .attr('class', 'u')

    g.append('text')
      .attr('fill', '#333')
      .text(function (d) { return d })

    var zoomListener = d3.behavior.zoom()
      .scaleExtent([0.1, 10])
      .center([0, 0])
      .on('zoom', zoomHandler)
    zoomListener(svg)
    d3.select(window).on('resize', resize)
    resize()
  }

  var tx = 0
  var ty = 0
  var ss = 1
  function zoomHandler () {
    tx = d3.event.translate[0]
    ty = d3.event.translate[1]
    ss = d3.event.scale
  }

  var stepnum = 0
  var tic = performance.now()
  function step () {
    console.time('step')
    var cost = T.step()
    console.timeEnd('step')
    var fps = Math.round((T.iter / (performance.now() - tic)) * 1000)
    $('#cost').html('iteration ' + T.iter + ', fps: ' + fps + ', cost: ' + cost)
    updateEmbedding()

    if (stepnum === 10) {
      if (DO_PROFILE && window.console && window.console.profile) {
        console.profileEnd()
      }
    }

    stepnum++
    requestAnimationFrame(step)
  }

  var DO_PROFILE = true

  $(window).load(function () {
    $.getJSON('/data/wordvecs50dtop1000.json', function (j) {
    //$.getJSON('/data/shortglove.json', function (j) {
      //data = j
      var N = 500
      data = {
        words: j.words.slice(0, N),
        vecs: j.vecs.slice(0, N),
      }

      if (DO_PROFILE && window.console && window.console.profile) {
        console.profile('initialization')
      }

      console.time('streamingInit')
      T.initData(data.vecs) // init embedding  WITH SUBSET
      console.timeEnd('streamingInit')

      if (window.console && window.console.profile) {
        console.profileEnd()
      }

      // compare with karpathy's tSNE
      var Tkarpathy = new tsnejs.tSNE()
      console.time('karpathyInit')
      Tkarpathy.initDataRaw(data.vecs)
      console.timeEnd('karpathyInit')
      /*
      const n = Math.sqrt(Tkarpathy.P.length);
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (Math.abs(Tkarpathy.P[i*n+j] - T.P.get(i, j)) > 1e-5) {
            console.log('bad', Tkarpathy.P[i*n+j], T.P.get(i, j))
          }
        }
      }
      */
      drawEmbedding() // draw initial embedding
      if (DO_PROFILE && window.console && window.console.profile) {
        console.profile('step')
      }
      requestAnimationFrame(step)

      $('#cost').click(function() {
        console.log('adding word', j.words[N])
        T.add(j.vecs[N])
        N++
        data.words = j.words.slice(0, N)
        d3.selectAll('.viewport > svg').remove()
        drawEmbedding()  // redraw?
      })
    })
  })
})(tsne, $, d3, performance, tsnejs)
