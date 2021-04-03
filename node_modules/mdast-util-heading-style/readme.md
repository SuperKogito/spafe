# mdast-util-heading-style

[![Build][build-badge]][build]
[![Coverage][coverage-badge]][coverage]
[![Downloads][downloads-badge]][downloads]
[![Size][size-badge]][size]
[![Sponsors][sponsors-badge]][collective]
[![Backers][backers-badge]][collective]
[![Chat][chat-badge]][chat]

[**mdast**][mdast] utility to get the style of a heading.

## Install

[npm][]:

```sh
npm install mdast-util-heading-style
```

## Usage

```js
var style = require('mdast-util-heading-style')
var remark = require('remark')()

style(remark.parse('# ATX').children[0]) // => 'atx'
style(remark.parse('# ATX #\n').children[0]) // => 'atx-closed'
style(remark.parse('ATX\n===').children[0]) // => 'setext'

style(remark.parse('### ATX').children[0]) // => null
style(remark.parse('### ATX').children[0], 'setext') // => 'setext'
```

## API

### `style(node[, relative])`

Get the heading style of a node.

###### Parameters

*   `node` ([`Node`][node]) — Node to parse
*   `relative` (`string`, optional) — Style to use for ambiguous headings
    (atx-headings with a level of three or more could also be setext)

###### Returns

`string` (`'atx'`, `'atx-closed'`, or `'setext'`) — When an ambiguous
heading is found, either `relative` or `null` is returned.

## Contribute

See [`contributing.md` in `syntax-tree/.github`][contributing] for ways to get
started.
See [`support.md`][support] for ways to get help.

This project has a [Code of Conduct][coc].
By interacting with this repository, organisation, or community you agree to
abide by its terms.

## License

[MIT][license] © [Titus Wormer][author]

<!-- Definitions -->

[build-badge]: https://img.shields.io/travis/syntax-tree/mdast-util-heading-style.svg

[build]: https://travis-ci.org/syntax-tree/mdast-util-heading-style

[coverage-badge]: https://img.shields.io/codecov/c/github/syntax-tree/mdast-util-heading-style.svg

[coverage]: https://codecov.io/github/syntax-tree/mdast-util-heading-style

[downloads-badge]: https://img.shields.io/npm/dm/mdast-util-heading-style.svg

[downloads]: https://www.npmjs.com/package/mdast-util-heading-style

[size-badge]: https://img.shields.io/bundlephobia/minzip/mdast-util-heading-style.svg

[size]: https://bundlephobia.com/result?p=mdast-util-heading-style

[sponsors-badge]: https://opencollective.com/unified/sponsors/badge.svg

[backers-badge]: https://opencollective.com/unified/backers/badge.svg

[collective]: https://opencollective.com/unified

[chat-badge]: https://img.shields.io/badge/join%20the%20community-on%20spectrum-7b16ff.svg

[chat]: https://spectrum.chat/unified/syntax-tree

[license]: license

[author]: https://wooorm.com

[npm]: https://docs.npmjs.com/cli/install

[contributing]: https://github.com/syntax-tree/.github/blob/master/contributing.md

[support]: https://github.com/syntax-tree/.github/blob/master/support.md

[coc]: https://github.com/syntax-tree/.github/blob/master/code-of-conduct.md

[mdast]: https://github.com/syntax-tree/mdast

[node]: https://github.com/syntax-tree/unist#node
