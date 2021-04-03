'use strict'

var control = require('unified-message-control')
var marker = require('mdast-comment-marker')
var xtend = require('xtend')

module.exports = messageControl

var test = [
  'html', // Comments are `html` nodes in mdast.
  'comment' // In MDX, comments have their own node.
]

function messageControl(options) {
  return control(xtend({marker: marker, test: test}, options))
}
