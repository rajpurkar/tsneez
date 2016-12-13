//
//  barnes-hut.js
//
//  implementation of the barnes-hut quadtree algorithm for n-body repulsion
//  http://www.cs.princeton.edu/courses/archive/fall03/cs126/assignments/barnes-hut.html
//
//  Created by Christian Swinehart on 2011-01-14.
//  Copyright (c) 2011 Samizdat Drafting Co. All rights reserved.
//

var BarnesHutTree = function(){
  var _branches = []
  var _branchCtr = 0
  var _root = null
  var _theta = .5

  var that = {
    init:function(topleft, bottomright, theta){
      _theta = theta

      // create a fresh root node for these spatial bounds
      _branchCtr = 0
      _root = that._newBranch()
      _root.origin = topleft
      _root.size = bottomright.subtract(topleft)
    },

    initWithData: function(data, theta) {
      // compute top left and top right based on data and call init
      var N = data.shape[0]
      var topleft = new Point(0, 0)
      var bottomright = new Point(0, 0)
      for (var i = 0; i < N; i++) {
        var x = data.get(i, 0)
        var y = data.get(i, 1)
        if (x < topleft.x) topleft.x = x
        if (y < topleft.y) topleft.y = y
        if (x > bottomright.x) bottomright.x = x
        if (y > bottomright.y) bottomright.y = y
      }
      that.init(topleft, bottomright, theta);
      // then insert all the points
      for (var i = 0; i < N; i++) {
        that.insert(data.get(i, 0), data.get(i, 1));
      }
    },
    isCorrect: function() {

      var queue = [_root]


      var recurse = function(pnode, node) {
        if (node===undefined){
          return true;
        }

        if (!(pnode === null || _contains(pnode, node))) {
          console.error(pnode, node)
          return false;
        }

        // Just a point, or all children also are correct
        return (!('origin' in node) ||
            (recurse(node, node.nw)
            && recurse(node, node.ne)
            && recurse(node, node.sw)
            && recurse(node, node.se)
            ))
      }

      return recurse(null, _root)
    },
    insert:function(x, y){
      // Create new particle of mass = 1 (this lets us keep a count of the points in a cell)
      var newParticle = new Particle(new Point(x, y), 1)

      // add a particle to the tree, starting at the current _root and working down
      var node = _root
      var queue = [newParticle]

      while (queue.length){
        var particle = queue.shift()
        var p_mass = particle._m || particle.m
        var p_quad = that._whichQuad(particle, node)

        // Increment total mass and update center
        node.mass += p_mass
        var mult1 = (node.mass - 1) / node.mass
        var mult2 = 1 / node.mass
        if (node.p) {
          node.p = node.p.multiply(mult1).add(particle.p.multiply(mult2))
        } else {
          node.p = particle.p
        }

        if (node[p_quad]===undefined){
          // slot is empty, just drop this node in and update the mass/c.o.m.
          node[p_quad] = particle
        }else if ('origin' in node[p_quad]){
          // slot contains a branch node,
          // keep iterating with the branch as our new root
          node = node[p_quad]

          // put particle back
          queue.unshift(particle)
        }else{
          // slot contains a particle, create a new branch (subdivide)
          // and recurse with both points in the queue now
          var branch_size = node.size.divide(2)
          var branch_origin = new Point(node.origin)
          if (p_quad[0]=='s') branch_origin.y += branch_size.y
          if (p_quad[1]=='e') branch_origin.x += branch_size.x

          // replace the previously particle-occupied quad with a new internal branch node
          var oldParticle = node[p_quad]
          node[p_quad] = that._newBranch()
          node[p_quad].origin = branch_origin
          node[p_quad].size = branch_size

          // Switch down into the new branch
          node = node[p_quad]

          //if (oldParticle.p.x===particle.p.x && oldParticle.p.y===particle.p.y){
          //  // prevent infinite bisection in the case where two particles
          //  // have identical coordinates by jostling one of them slightly
          //  var x_spread = branch_size.x*.08
          //  var y_spread = branch_size.y*.08
          //  oldParticle.p.x = Math.min(branch_origin.x+branch_size.x,
          //                             Math.max(branch_origin.x,
          //                                      oldParticle.p.x - x_spread/2 +
          //                                      Math.random()*x_spread))
          //  oldParticle.p.y = Math.min(branch_origin.y+branch_size.y,
          //                             Math.max(branch_origin.y,
          //                                      oldParticle.p.y - y_spread/2 +
          //                                      Math.random()*y_spread))
          //}

          // keep iterating but now having to place both the current particle and the
          // one we just replaced with the branch node
          queue.push(oldParticle)
          queue.unshift(particle)
        }

      }

    },

    computeForces:function(x, y){
      // Compute forces on point at x
      // Create new particle of zero
      var particle = new Particle(new Point(x, y), 1)

      var qsum = 0.  // Z
      var count = 0

      var queue = [_root]
      while (queue.length){
        node = queue.shift()
        if (node===undefined) continue
        if (particle===node) continue

        if ('f' in node){
          count++
          // this is a particle leafnode, so just apply the force directly
          var d = particle.p.subtract(node.p);
          var aff = 1. / (1. + d.magnitudeSquared())
          var force = - aff * aff  // -qu_ij^2 = -(q_ij * Z)^2
          if (d.magnitude() < 1e-5) {
            continue
          }
          particle.applyForce(d.multiply(force))
          qsum += aff
        }else{
          // it's a branch node so decide if it's cluster-y and distant enough
          // to summarize as a single point. if it's too complex, open it and deal
          // with its quadrants in turn
          var dist = particle.p.subtract(node.p).magnitude()
          var rcell = Math.max(node.size.x, node.size.y)
          if (rcell/dist > _theta){ // i.e., s/d > Θ
            // open the quad and recurse
            queue.push(node.ne)
            queue.push(node.nw)
            queue.push(node.se)
            queue.push(node.sw)
          }else{
            // treat the quad as a single body
            var d = particle.p.subtract(node.p);
            var aff = 1. / (1. + d.magnitudeSquared())
            var force = - node.mass * aff * aff  // - N_cell * (q_{i,cell} * Z)^2
            var direction = (d.magnitude()>0) ? d : Point.random(1e-5)
            particle.applyForce(direction.multiply(force));
            qsum += node.mass * aff
            count += node.mass
          }
        }
      }

      // Return accumulated forces on the particle
      return {
        x: particle.f.x,
        y: particle.f.y,
        Z: qsum,
        count: count
      }
    },

    _whichQuad:function(particle, node){
      // sort the particle into one of the quadrants of this node
      if (particle.p.exploded()) return null
      var particle_p = particle.p.subtract(node.origin)
      var halfsize = node.size.divide(2)
      if (particle_p.y < halfsize.y){
        if (particle_p.x < halfsize.x) return 'nw'
        else return 'ne'
      }else{
        if (particle_p.x < halfsize.x) return 'sw'
        else return 'se'
      }
    },

    _newBranch:function(){
      // to prevent a gc horrorshow, recycle the tree nodes between iterations
      if (_branches[_branchCtr]){
        var branch = _branches[_branchCtr]
        branch.ne = branch.nw = branch.se = branch.sw = undefined
        branch.mass = 0
        delete branch.p
      }else{
        branch = {origin:null, size:null,
                  nw:undefined, ne:undefined, sw:undefined, se:undefined, mass:0}
        _branches[_branchCtr] = branch
      }

      _branchCtr++
      return branch
    }
  }

  return that
}

var _contains = function(node, child) {
    var bottomright = node.origin.add(node.size)
    return (
        child.p.x >= node.origin.x - 1e-5
        && child.p.y >= node.origin.y - 1e-5
        && child.p.x <= bottomright.x + 1e-5
        && child.p.y <= bottomright.y + 1e-5
        )
}

var Particle = function(position, mass){
  this.p = position;
  this.m = mass;
	this.v = new Point(0, 0); // velocity
	this.f = new Point(0, 0); // force
};
Particle.prototype.applyForce = function(force){
	this.f = this.f.add(force);
};

var Point = function(x, y){
  if (x && x.hasOwnProperty('y')){
    y = x.y; x=x.x;
  }
  this.x = x;
  this.y = y;
}

Point.random = function(radius){
  console.error('random??')
  radius = (radius!==undefined) ? radius : 5
	return new Point(2*radius * (Math.random() - 0.5), 2*radius* (Math.random() - 0.5));
}

Point.prototype = {
  exploded:function(){
    return ( isNaN(this.x) || isNaN(this.y) )
  },
  add:function(v2){
  	return new Point(this.x + v2.x, this.y + v2.y);
  },
  subtract:function(v2){
  	return new Point(this.x - v2.x, this.y - v2.y);
  },
  multiply:function(n){
  	return new Point(this.x * n, this.y * n);
  },
  divide:function(n){
  	return new Point(this.x / n, this.y / n);
  },
  magnitude:function(){
  	return Math.sqrt(this.x*this.x + this.y*this.y);
  },
  magnitudeSquared:function(){
  	return this.x*this.x + this.y*this.y;
  },
  normal:function(){
  	return new Point(-this.y, this.x);
  },
  normalize:function(){
  	return this.divide(this.magnitude());
  }
}

module.exports = {
  BarnesHutTree: BarnesHutTree,
  Point: Point,
  Particle: Particle,
}
