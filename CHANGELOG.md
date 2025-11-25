# Changelog

## [1.3.3](https://github.com/chanzuckerberg/octopi/compare/octopi-v1.3.2...octopi-v1.3.3) (2025-11-18)


### 🐞 Bug Fixes

* avoid modifying original objects while iterating ([#46](https://github.com/chanzuckerberg/octopi/issues/46)) ([c1586de](https://github.com/chanzuckerberg/octopi/commit/c1586de2c7f833aea57c74952980ca77686b8bab))

## [1.3.2](https://github.com/chanzuckerberg/octopi/compare/octopi-v1.3.1...octopi-v1.3.2) (2025-11-16)


### 🐞 Bug Fixes

* ensure filtering of picks is correct ([#44](https://github.com/chanzuckerberg/octopi/issues/44)) ([8a70056](https://github.com/chanzuckerberg/octopi/commit/8a700566fba5104b4d16acd98cabced6f07db624))

## [1.3.1](https://github.com/chanzuckerberg/octopi/compare/octopi-v1.3.0...octopi-v1.3.1) (2025-11-15)


### 🐞 Bug Fixes

* keep sw parameter for vcp ([#42](https://github.com/chanzuckerberg/octopi/issues/42)) ([41f5080](https://github.com/chanzuckerberg/octopi/commit/41f50800b32d96abaf1ff64bcaad99c7cb179f15))

## [1.3.0](https://github.com/chanzuckerberg/octopi/compare/octopi-v1.2.0...octopi-v1.3.0) (2025-11-13)


### ✨ Features

* logging ([#26](https://github.com/chanzuckerberg/octopi/issues/26)) ([7ff5e72](https://github.com/chanzuckerberg/octopi/commit/7ff5e723796d8a8ef3fdb46f8f07fa7f527b6a11))
* Optimized inference ([#39](https://github.com/chanzuckerberg/octopi/issues/39)) ([c53a3aa](https://github.com/chanzuckerberg/octopi/commit/c53a3aaf08d5197395d7c6e1fb01bc447c4a9b76))
* remove repeating picks ([#37](https://github.com/chanzuckerberg/octopi/issues/37)) ([aea7d4e](https://github.com/chanzuckerberg/octopi/commit/aea7d4eb2572bce679584cbbab8310de46622163))


### 🐞 Bug Fixes

* allow users to write multiple segmentation targets ([#25](https://github.com/chanzuckerberg/octopi/issues/25)) ([e19509f](https://github.com/chanzuckerberg/octopi/commit/e19509f322f53fde23c8e597ee00b549f7034874))
* beta test feedback ([#27](https://github.com/chanzuckerberg/octopi/issues/27)) ([fab47b8](https://github.com/chanzuckerberg/octopi/commit/fab47b89c31971f246c82ca6d92acd89429bf438))
* minor fixes to writing of new picks and generating segmentation targets ([#24](https://github.com/chanzuckerberg/octopi/issues/24)) ([7a9b7af](https://github.com/chanzuckerberg/octopi/commit/7a9b7af535f620d636319f30428b3edd838f1195))
* nas optimization ([#36](https://github.com/chanzuckerberg/octopi/issues/36)) ([29e18a1](https://github.com/chanzuckerberg/octopi/commit/29e18a15137b6acce899d72f6c8f96fe3a0552d9))
* standard output for log files ([#38](https://github.com/chanzuckerberg/octopi/issues/38)) ([c867efc](https://github.com/chanzuckerberg/octopi/commit/c867efcb46d5f0425fad0136e9d79e9b7d76025b))
* visualization ([#40](https://github.com/chanzuckerberg/octopi/issues/40)) ([812c103](https://github.com/chanzuckerberg/octopi/commit/812c1039bcf9d9cd551592de344adac97f81aa6b))


### 📝 Documentation

* Update README to reflect pip 25.2 updates ([#22](https://github.com/chanzuckerberg/octopi/issues/22)) ([3cd6b4c](https://github.com/chanzuckerberg/octopi/commit/3cd6b4ce1b2145b5c474eee9e3f1fcabb384df3e))

## [1.2.0](https://github.com/chanzuckerberg/octopi/compare/octopi-v1.1.0...octopi-v1.2.0) (2025-08-07)


### ✨ Features

* add release-please configuration for automated releases ([#13](https://github.com/chanzuckerberg/octopi/issues/13)) ([afc2a49](https://github.com/chanzuckerberg/octopi/commit/afc2a49e972c6f2225d01b58712d9bb1da6673db))
* model soup ([#20](https://github.com/chanzuckerberg/octopi/issues/20)) ([81e45c7](https://github.com/chanzuckerberg/octopi/commit/81e45c7b2a3422132d936ad04ba971d8e16eef90))


### 🐞 Bug Fixes

* Add docs extra for docs dependencies. ([#16](https://github.com/chanzuckerberg/octopi/issues/16)) ([14a5b65](https://github.com/chanzuckerberg/octopi/commit/14a5b655050529d5094d3eb2ebd48856ebfe2c34))
* dynamic versioning ([#21](https://github.com/chanzuckerberg/octopi/issues/21)) ([def60ab](https://github.com/chanzuckerberg/octopi/commit/def60ab403b9c8a7b8bd5a51b8f9c079a85b38e1))
* improvements in cli experience and localize command ([52ee7ac](https://github.com/chanzuckerberg/octopi/commit/52ee7ac2f001ffbb096fac320ab9610ca1673855))
* setup automated releases with organization secrets ([#15](https://github.com/chanzuckerberg/octopi/issues/15)) ([a3b04cf](https://github.com/chanzuckerberg/octopi/commit/a3b04cfca8d161953b310b2a18a61995628734ec))


### 🧹 Miscellaneous Chores

* CCIE-4984 conform to open sourcing guidelines ([#19](https://github.com/chanzuckerberg/octopi/issues/19)) ([bd59cde](https://github.com/chanzuckerberg/octopi/commit/bd59cdeee4e85d6bc29e7d4151885e9a9344e508))


### ♻️ Code Refactoring

* Modularize utilities for parsing, I/O, and configuration ([#17](https://github.com/chanzuckerberg/octopi/issues/17)) ([ff54f15](https://github.com/chanzuckerberg/octopi/commit/ff54f1559077a1ef5655fb2d34c7eaf328be2caf))
