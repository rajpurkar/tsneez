(function (tsne, $, d3, performance, tsnejs) {

  $(function () {
    var T = new tsne.TSNE({
      theta: 0.8
    }) // create a tSNE instance

    var vecs
    var words
    var svg
    $.getJSON('/data/wordvecs50dtop1000.json', function (dataAll) {
      var spliceIndex = 100
      vecs = dataAll.vecs.splice(0, spliceIndex)
      words = dataAll.words.splice(0, spliceIndex)
      T.initData(vecs)
      setInterval(function () {
        var vec = dataAll.vecs.splice(0, 1)[0]
        var word = dataAll.words.splice(0, 1)[0]
        T.add(vec)
        words.push(word)
        // updateEmbedding()
      }, 3000) 
      var div = d3.select('.viewport')
      svg = div.append('svg') // svg is global
      var zoomListener = d3.behavior.zoom()
        .scaleExtent([0.1, 10])
        .center([0, 0])
        .on('zoom', zoomHandler)
      zoomListener(svg)
      d3.select(window).on('resize', resize)
      resize()
      d3.timer(step)
    })

    function updateEmbedding () {
      var g = svg.selectAll('.b')
        .data(words)
        .enter().append('g')
        .attr('class', 'u')

      g.append('text')
        .attr('fill', '#333')
        .text(function (d) { return d })

      var Y = T.Y
      svg.selectAll('.u')
      .data(words)
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

    var tx = 0
    var ty = 0
    var ss = 1

    function zoomHandler () {
      tx = d3.event.translate[0]
      ty = d3.event.translate[1]
      ss = d3.event.scale
    }

    function step () {
      T.step()
      updateEmbedding()
    }
  })
})(tsne, $, d3, performance, tsnejs)
