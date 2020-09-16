# Name/Word generation using Markov chains

![Rust](https://github.com/mikeando/markovian/workflows/Rust/badge.svg?branch=master)


# What is markovian?

Markovian is a command line utility for generating fake words or names.
You give it one or more lists of words and it analyses their structure,
finding some of the underlying rules and then using those to generate new words.
Not all the words it spits out will be good, and some will be real
words or names rather than fake ones. It may even decide to give you profanities...

# Getting markovia

## Precompiled releases
The easiest way is to download one of the precompiled versions from the releases page
https://github.com/mikeando/markovian/releases.

If you're using windows you'll need to have the microsoft Visual C++ Redistributable libraries 
installed, which can be downloaded from https://www.microsoft.com/en-au/download/details.aspx?id=48145 

There is a build for linux, but it is currently untested - I don't expect any issues, but it may not work at all.


# Simple usage

```
$ markovian simple generate --encoding=string resources/Moby_Names_M_lc.txt
stergiramessey
barnan
bralph
jerrel
jord
...
```

# How it works

Every word in the input list is broken down into pieces, and then all triplets of these
pieces are recorded. Two additional start pieces are appended at the start and end of every word.
We'll denote these as `^` an `$` respectively.

For example the word "CAT" would first be broken into the pieces `^`, `^`, `C`, `A`, `T`, `$`, `$`.
Then the triplets would be stored.

```
^^C
^CA
CAT
AT$
T$$
```

Generating a new word begins by looking for all triplets matching `^^?` and picking one. We then
repeatedly take the most recent 2 parts and look for a matching triplet until we hit an end symbol

For example if we've added just the words CAT and DOG and DOLL to our lists, we might end up regenerating the word dog as follows

```
      ^^? 
          - we have ^^C or ^^D  - pick ^^D
^     ^D?
          - both DOG and DOLL contribute ^DO, there are no other options
^^    DO?
          - DOG contribites DOG, DOLL contributes DOL - pick DOG
^^D   OG? 
          - only DOG contributes a matching triplet, OG$
^^DOG$?
```

This is a fairly standard Markov chain generation. Markovian tries to be a little bit smarter, by 
treating some sequences of characters as a single unit. For example the names joseph and philip might be 
treated as 

```
j|o|se|ph
ph|i|l|ip
```

resulting in triplets like:

```
j|o|se
o|se|ph
...
```

# Advanced usage

Generation from scratch can take a while on large word lists, 
so markovian can save the intermediate parts, leading to a final 
file called a generator that can be used to quickly generate words.

Conceptually going from a word list to a generator has three steps.

1. Determining the input symbols.
2. Identifying and combining symbols that occur together.
3. Building triplet map

## Generating the initial symbol table

The first stage is 

```
markovian symbol-table generate --encoding=string --output=A.symboltable --input=word-list-1.txt --input=word-list-2.txt
```

This generates a symbol table file called `A.symboltable` containing all the symbols from the two input word lists --- 
you'll want to use a better name for the output.

You can see the list of symbols it uses 

```
markovian symbol-table print --input=A.symtable
```

For example 

```
> markovian symbol-table generate --encoding=string --output=Moby_initial.symboltable --input=resources/Moby_Names_M_lc.txt 
using 3878 input strings
found 30 symbols
wrote Moby_initial.symboltable 
> markovian symbol-table print --input=Moby_initial.symboltable
encoding: char
max symbol id: 30
0 => START
1 => END
2 => a
3 => r
4 => o
5 => n
...
24 => x
25 => z
26 => v
27 => '
28 =>  
29 => q
```

## Combining symbols

This step works on an existing symbol table file and looks for symbols that occur together frequently in the input 
and combines them into one compound symbol.

```
markovian symbol-table improve A.symboltable --output=B.symboltable word-list-1.txt word-list-2.txt
```

For example 

```
> markovian symbol-table improve Moby_initial.symboltable --output Moby_50.symboltable resources/Moby_Names_M_lc.txt 
...
> markovian symbol-table print --input=Moby_50.symboltable
encoding: char
max symbol id: 80
0 => START
1 => END
2 => a
3 => r
...
29 => q
30 => er
31 => ar
...
75 => em
76 => ab
77 => do
```

We can then see how this symbol-table breaks up words using

```
> markovian symbol-table symbolify --symbol-separator="." Moby_50.symboltable johnathon stephan arnold eric
johnathon => ["j.o.h.n.a.th.on"]
stephan => ["st.e.p.h.an"]
arnold => ["ar.n.ol.d"]
eric => ["er.i.c", "e.ri.c"]
```

We only show the shortest symbols that produce the given word, but it is possible that more than one combination
can produce the same length - in the example above `eric` can be written two ways.


At the moment this performs a fixed number (50) of symbol combining steps. This will become configurable in the future.

If you want to combine more symbols you can rerun this stage on the new symbol table file.

## Generating the triplet maps / generator

We create the triplet maps / generator file using

```
> markovian generator create B.symboltable --output=A.generator word-list-1.txt word-list-2.txt
```

## Generating words with the generator

```
markovian generator generate A.generator 
```

## Comparison between amounts of combining

Here we have the results of no combining, 1 combining run (50 steps), 2 combining runs (100 steps) and 3 combining runs (150 steps)


```
None            50            100         150           
----            ----          ----        ----
derrikosmenie   abe           magnum      esterby
cultombry       lo            pincey      valentissiley
ruy             alfie         elvin       cord
reiffre         tadufenton    gelino      terrentin
rold            lus           peathar     jacouve
roddinionie     timothy       arlando     hill
do              burle         orton       gaylen
guig            silvern       mery        levi
bard            kellen        vacques     nestofforten
cartie          jave          das         lymann
my              pattie        justy       gibby
valud           ham           von         flie
denobb          mayn          gusse       dolain
ke              vin           mon         wallie
lannes          clard         valleard    suthrie
ammy            carris        tan         pin
wal             erthorthray   erin        farr
die             peit          fair        mack
sigiart         humashan      fran        bogart
abaley          santons       ter         ripporsigis
```

I would claim that the results from the 50, 100 and 150 columns are 
in general better names than those from the None column. 
But I'm not sure that I could quantify this in any meaningful way.

# 3rd party components

Building an application often uses a lot of third-party parts.

This code directly uses the following libraries:

## Directly included libraries licensed under the MIT license

The MIT License is:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The libraries with versions and copyright holders covered by this license are:

* bincode (v1.3.1) https://github.com/servo/bincode : Copyright (c) 2014 Ty Overby
* fern (v0.6.0) https://github.com/daboross/fern : Copyright (c) 2014-2017 David Ross
* log (v0.4.8) : https://github.com/rust-lang/log : Copyright (c) 2014 The Rust Project Developers
* rand (v0.7.3) : https://github.com/rust-random/rand : Copyright 2018 Developers of the Rand project, Copyright (c) 2014 The Rust Project Developers
* serde (v1.0.113) : https://github.com/serde-rs/serde
* snafu (v0.6.8) : https://github.com/shepmaster/snafu : Copyright (c) 2019- Jake Goulding
* structopt (v0.3.12) : https://github.com/TeXitoi/structopt : Copyright (c) 2018 Guillaume Pinot (@TeXitoi) <texitoi@texitoi.eu>

## Indirect 3rd party code under MIT License

These are libraries that the above directly included libraries may include. 
Code from these libraries may make it into executables.

* ansi-term (v0.11.0) https://github.com/ogham/rust-ansi-term : Copyright (c) 2014 Benjamin Sago
* atty (v0.2.14) https://github.com/softprops/atty : Copyright (c) 2015-2019 Doug Tangren
* bitflags (v1.2.1) https://github.com/bitflags/bitflags : Copyright (c) 2014 The Rust Project Developers
* byteorder (v1.3.4) https://github.com/BurntSushi/byteorder : Copyright (c) 2015 Andrew Gallant
* cfg-if (v0.1.10) https://github.com/alexcrichton/cfg-if : Copyright (c) 2014 Alex Crichton
* clap (v2.33.0) https://github.com/clap-rs/clap : Copyright (c) 2015-2016 Kevin B. Knapp
* doc-comment (v0.3.3) https://github.com/GuillaumeGomez/doc-comment : Copyright (c) 2018 Guillaume Gomez
* getrandom (v0.1.14) https://github.com/rust-random/getrandom : Copyright 2018 Developers of the Rand project, Copyright (c) 2014 The Rust Project Developers
* heck (v0.3.1) https://github.com/withoutboats/heck : Copyright (c) 2015 The Rust Project Developers
* hermit-abi (v0.1.10) https://github.com/hermitcore/rusty-hermit
* lazy_static (v1.4.0) https://github.com/rust-lang-nursery/lazy-static.rs : Copyright (c) 2010 The Rust Project Developers
* libc (v0.2.72) https://github.com/rust-lang/libc : Copyright (c) 2014-2020 The Rust Project Developers
* ppv-lite86 (v0.2.6) https://github.com/cryptocorrosion/cryptocorrosion : Copyright (c) 2019 The CryptoCorrosion Contributors
* proc-macro-error (v0.4.12) https://gitlab.com/CreepySkeleton/proc-macro-error : Copyright (c) 2019-2020 CreepySkeleton
* proc-macro2 (v1.0.10) https://github.com/alexcrichton/proc-macro2 : Copyright (c) 2014 Alex Crichton
* quote (v1.0.3) https://github.com/dtolnay/quote : Copyright (c) 2016 The Rust Project Developers
* strsim (v0.8.0) https://github.com/dguo/strsim-rs : Copyright (c) 2015 Danny Guo, Copyright (c) 2016 Titus Wormer <tituswormer@gmail.com>, Copyright (c) 2018 Akash Kurdekar
* syn (v1.0.17) https://github.com/dtolnay/syn
* syn-mid (v0.5.0) https://github.com/taiki-e/syn-mid
* textwrap (v0.11.0) https://github.com/mgeisler/textwrap: Copyright (c) 2016 Martin Geisler
* unicode-segmentation (v1.6.0) https://github.com/unicode-rs/unicode-segmentation : Copyright (c) 2015 The Rust Project Developers
* unicode-width (v0.1.7) https://github.com/unicode-rs/unicode-width : Copyright (c) 2015 The Rust Project Developers
* unicode-xid (v0.2.0) https://github.com/unicode-rs/unicode-xid : Copyright (c) 2015 The Rust Project Developers
* vec_map (v0.8.1) https://github.com/contain-rs/vec-map : Copyright (c) 2015 The Rust Project Developers
* version_check (v0.9.1) https://github.com/SergioBenitez/version_check : Copyright (c) 2017-2018 Sergio Benitez
* wasi (v0.9.0+wasi-snapshot-preview1) https://github.com/bytecodealliance/wasi
* winapi (v0.3.8) https://github.com/retep998/winapi-rs : Copyright (c) 2015-2018 The winapi-rs Developers


## Indirect 3rd party code under MPL-2 License

* colored (v1.9.3) https://github.com/mackwic/colored







