var N = 100
var GRADIENT_STEPS = 50
var stepnum = 0


// Hyper parameters
var opt = {}
opt.theta = 0.5 // theta is ...
opt.perplexity = 10 // perplexity is ...

var stepEmbedding, getEmbedding
var tsneez = new tsneez.TSNEEZ(opt) // create a tsneez instance
stepEmbedding = function () { stepnum++; return tsneez.step() }
getEmbedding = function () { return tsneez.Y }

/* 
    applies tsneez algorithm to the vectors
    returns corresponding two-dimensional vectors
*/
function dimensionReduce(vecs) {
    // Data mapped to d3 format for visualization
    vecs.words = vecs.words.map(function (word) {
        return {str: String(word), init: true}
    })

    var vecs = {
        words: vecs.words.slice(0, N),
        vecs: vecs.vecs.slice(0, N)
    }

    tsneez.initData(vecs.vecs)

    drawEmbedding(vecs)

    var loopfn = function () {
        stepEmbedding()
        console.log(stepnum)
        if (stepnum === 100) return
        updateEmbedding(vecs.words, getEmbedding())
        window.requestAnimationFrame(loopfn)
    }
    window.requestAnimationFrame(loopfn)

    // for(var k = 0; k < GRADIENT_STEPS; k++) {
    //     console.log(`Step : ${k}`)
    //     stepEmbedding() // gradient update
    //     var Y = getEmbedding()
    //     updateEmbedding(vecs.words, Y)
    // }
    // visEmbedding(vecs)
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
function draw (data) {
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

  // Resize the viewport in response to window resize
function resize () {
    var width = $('.viewport').width()
    var height = 600
    svg.attr('width', width).attr('height', height)
  }

// Draw initial embedding
function drawEmbedding (data) {
    var div = d3.select('.viewport')
    svg = div.append('svg') // svg is global
    draw(data)
    zoomListener(svg)
    d3.select(window).on('resize', resize)
    resize()
}

// Update d3 embedding on a step
function updateEmbedding (words, Y) {
    if (Y === null) return  // scienceai might not be ready
    var s = svg.selectAll('.u')
    .data(words)
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

function visEmbedding(data) {
    console.log(data)
    updateEmbedding(data)
    window.requestAnimationFrame(visEmbedding(data))
}


// Fetch data from a json file.
var shortgloveFile = '/tsneez/data/shortglove.json'

document.addEventListener('DOMContentLoaded', function() {
    fetch(shortgloveFile)
        .then(data => data.json())
        .then((vecs) => {
            dimensionReduce(vecs)
    });
});