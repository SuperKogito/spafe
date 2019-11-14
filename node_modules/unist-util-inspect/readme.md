# unist-util-inspect

[![Build][build-badge]][build]
[![Coverage][coverage-badge]][coverage]
[![Downloads][downloads-badge]][downloads]
[![Size][size-badge]][size]
[![Sponsors][sponsors-badge]][collective]
[![Backers][backers-badge]][collective]
[![Chat][chat-badge]][chat]

[**unist**][unist] utility to inspect nodes.

## Install

[npm][]:

```sh
npm install unist-util-inspect
```

## Usage

```js
var u = require('unist-builder')
var inspect = require('unist-util-inspect')

var tree = u('root', [
  u('literal', '1'),
  u('parent', [
    u('void', {id: 'a'}),
    u('literal', '2'),
    u('node', {id: 'b'}, [])
  ])
])

console.log(inspect(tree))
```

Yields:

```text
root[2]
├─ literal: "1"
└─ parent[3]
   ├─ void [id="a"]
   ├─ literal: "2"
   └─ node[0] [id="b"]
```

## API

### `inspect(node)`

By default, color support is enabled on Node.js and turned off anywhere else.
See below on how to change that.

###### Parameters

*   `node` ([`Node`][node]).

###### Returns

`string` — String representing `node`.

### `inspect.<style>[.<style>…](node)`

Where `<style>` is either `color` or `noColor`.

To explicitly add or remove ANSI sequences, use either `inspect.color(node)`
or `inspect.noColor(node)`.

## Contribute

See [`contributing.md` in `syntax-tree/.github`][contributing] for ways to get
started.
See [`support.md`][support] for ways to get help.

This project has a [Code of Conduct][coc].
By interacting with this repository, organisation, or community you agree to
abide by its terms.

## License

[MIT][license] © [Titus Wormer][author]

<!-- Definition -->

[build-badge]: https://img.shields.io/travis/syntax-tree/unist-util-inspect.svg

[build]: https://travis-ci.org/syntax-tree/unist-util-inspect

[coverage-badge]: https://img.shields.io/codecov/c/github/syntax-tree/unist-util-inspect.svg

[coverage]: https://codecov.io/github/syntax-tree/unist-util-inspect

[downloads-badge]: https://img.shields.io/npm/dm/unist-util-inspect.svg

[downloads]: https://www.npmjs.com/package/unist-util-inspect

[size-badge]: https://img.shields.io/bundlephobia/minzip/unist-util-inspect.svg

[size]: https://bundlephobia.com/result?p=unist-util-inspect

[sponsors-badge]: https://opencollective.com/unified/sponsors/badge.svg

[backers-badge]: https://opencollective.com/unified/backers/badge.svg

[collective]: https://opencollective.com/unified

[chat-badge]: https://img.shields.io/badge/join%20the%20community-on%20spectrum-7b16ff.svg

[chat]: https://spectrum.chat/unified/syntax-tree

[unist]: https://github.com/syntax-tree/unist

[npm]: https://docs.npmjs.com/cli/install

[node]: https://github.com/syntax-tree/unist#node

[license]: license

[author]: https://wooorm.com

[contributing]: https://github.com/syntax-tree/.github/blob/master/contributing.md

[support]: https://github.com/syntax-tree/.github/blob/master/support.md

[coc]: https://github.com/syntax-tree/.github/blob/master/code-of-conduct.md
