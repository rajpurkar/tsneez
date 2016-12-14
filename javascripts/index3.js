(function (tsne, $, d3, performance, tsnejs) {
  var T = new tsne.TSNE({
    theta: 0.8
  }) // create a tSNE instance
  var data

  function updateEmbedding () {
    var text = svg.selectAll('text')
    .data(data.words, function (d) { return d })

    text.enter().append('text')
      .attr('class', 'enter')
      .text(function (d) { return d })

    var Y = T.Y
    text.attr('transform', function (d, i) {
      return 'translate(' +
        ((Y.get(i, 0) * 200 * ss + tx) + 400) + ',' +
        ((Y.get(i, 1) * 200 * ss + ty) + 400) + ')'
    })
  }

  function resize () {
    var width = $('.viewport').width()
    var height = 400
    svg.attr('width', width).attr('height', height)
  }

  var svg

  var zoomListener = d3.behavior.zoom()
    .scaleExtent([0.1, 10])
    .center([0, 0])
    .on('zoom', zoomHandler)

  d3.select(window).on('resize', resize)

  function drawEmbedding () {
    var div = d3.select('.viewport')

    svg = div.append('svg') // svg is global
    /*
    var g = svg.selectAll('.b')
      .data(data.words)
      .enter().append('g')
      .attr('class', 'u')

    g.append('text')
      .attr('fill', '#333')
      .text(function (d) { return d })
    */
    zoomListener(svg)
    resize()
  }

  function updateUp () {
    var g = svg.selectAll('.b')
      .data(data.words)
      .enter().append('g')
      .attr('class', 'u')

    g.append('text')
      .attr('fill', '#333')
      .text(function (d) { return d })
  }

  var tx = 0
  var ty = 0
  var ss = 1
  function zoomHandler () {
    tx = d3.event.translate[0]
    ty = d3.event.translate[1]
    ss = d3.event.scale
  }

  var tic = performance.now()
  function step () {
    var cost = T.step()
    var fps = Math.round((T.iter / (performance.now() - tic)) * 1000)
    $('#cost').html('iteration ' + T.iter + ', fps: ' + fps + ', cost: ' + cost)
    updateEmbedding()
    requestAnimationFrame(step)
  }

 $(window).load(function () {
    $.getJSON('/data/wordvecs50dtop1000.json', function (j) {
      var N = 800
      data = {
        words: j.words.slice(0, N),
        vecs: j.vecs.slice(0, N)
      }
      T.initData(data.vecs) // init embedding  WITH SUBSET
      drawEmbedding() // draw initial embedding
      requestAnimationFrame(step)

      $('#cost').click(function () {
        console.log('adding word', j.words[N])
        T.add(j.vecs[N])
        N++
        data.words = j.words.slice(0, N)
        d3.selectAll('.viewport > svg').remove()
        drawEmbedding()  // redraw?
        //updateUp()
      })
    })
  })
})(tsne, $, d3, performance, tsnejs)
