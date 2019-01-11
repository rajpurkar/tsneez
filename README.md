# tsneez
## High Dimensional Visualization Simplified

### Example 1: Data passed as high dimensional pairwise dissimilar vectors
Import tsneez.js into your document: <script src='/tsneez/dist/tsneez.js'></script> and then is some of the usage code.

```javascript
// Hyper parameters
var opt = {}
opt.theta = 0.5 // theta
opt.perplexity = 10 // perplexity
var GRADIENT_STEPS = 5000 


var tsneez = new tsneez.TSNEEZ(opt) // create a tsneez instance

// Initialize data. We have four high dimensional pairwise dissimilar vectors
var vects = [
    [0.418, 0.24968, -0.41242, 0.1217, 0.34527],
    [0.013441, 0.23682, -0.16899, 0.40951, 0.63812],
    [0.15164, 0.30177, -0.16763, 0.17684, 0.31719],
    [0.70853, 0.57088, -0.4716, 0.18048, 0.54449],
]
tsneez.initData(vecs)

// A function that applies tsneez algorithm to reduce high dimenionsal vectors
function dimensionReduce(vecs) {

    tsneez.initData(vecs)

    for(var k = 0; k < GRADIENT_STEPS ; k++) {
        tsneez.step() // gradient update
        console.log(`Step : ${k}`)
    }

    var Y = tsneez.Y 

    return Y // Y is an array of 2-dimensional vectors that you can plot.
}

// Execute dimensionality reduction
dimensionReduce(vecs)
```

# Example 2: Data passed to tsneez as a set of high-dimensional Glove vectors
Append the following code within the scripts (<script></script>)tags of your document.

```javascript
// Fetch data from a json file of Glove vectors and execute dimenisonal reductionality
var shortgloveFile = '/tsneez/data/shortglove.json' // A json file of Glove vectors.
document.addEventListener('DOMContentLoaded', function() {
    fetch(shortgloveFile) 
        .then(data => data.json())
        .then((gloves) => {
            var vecs = gloves["vecs"]
            dimensionReduce(vecs)
    })
})
```

# Example 3: tsneez visualization
(a) Create a HTML document to view the visualization.

```javascript
<!DOCTYPE html>
<html>
    <head>
        <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
        <script src="/tsneez/bower_components/d3/d3.min.js"></script>
        <script src="/tsneez/bower_components/randomcolor/randomColor.js"></script>
        
        <script src="/tsneez/dist/tsneez.js"></script>
        <script src='./example3.js'></script>
        
        <title>tsneez Visualization</title>
        
        <style>
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 10px;
                background-color:whitesmoke;
            }
            .viewport {
                width: 800px;
                height: 600px;
            }
            .viewport svg {
                width: 800px;
                height: 600px;
                display: block;
            }
            .viewport svg text {
                pointer-events: none;
                font-family: 'Lato', Arial;
                font-weight: 500;
                font-size: 1em;
                fill: #444;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row">
                <div class="viewport">
                </div>
            </div>
        </div>
    </body>
</html>
```

(b) Create an example3.js javascript file in the same directory as that of the HTML document and append the following code.
```javascript
// Hyper parameters
var opt = {}
opt.theta = 0.9 // theta
opt.perplexity = 20 // perplexity

function dimensionReduce(vecs) {
    var NUMBER_OF_EMBEDDINGS = 1000
    tsneez.initData(vecs.vecs)
    visEmbedding(vecs, NUMBER_OF_EMBEDDINGS)
}

// Fetch data from a json file.
var shortgloveFile = '/tsneez/data/shortglove.json'
document.addEventListener('DOMContentLoaded', function() {
    fetch(shortgloveFile)
        .then(data => data.json())
        .then((vecs) => {
            dimensionReduce(vecs)
    })
})


// define and create a tsneez instance
var stepnum = 0
var stepEmbedding, getEmbedding
var tsneez = new tsneez.TSNEEZ(opt) // create a tsneez instance
initData = function (vecs) { tsneez.initData(vecs) }
stepEmbedding = function () { stepnum++; return tsneez.step() }
getEmbedding = function () { return tsneez.Y }
var visEmbedding = function(vecs, number_of_embeddings) {
    function convertToD3Format(v) {
        let converted = Object.assign({}, v)
        converted.words = v.words.map(function (word) {
            return { str: String(word), init: true,}
        })
        converted = {
            words: converted.words.slice(0,number_of_embeddings),
            vecs: converted.vecs.slice(0,number_of_embeddings),
        }
        return converted
    }

    function animate() {
        stepEmbedding()
        updateEmbedding(converted.words)
        window.requestAnimationFrame(animate)
    }

    let converted = convertToD3Format(vecs)
    drawEmbedding(converted)
    window.requestAnimationFrame(animate) 
}

// Set up visualization
var svg 
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
function updateEmbedding (words)  {
    var Y = getEmbedding()
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
}
```