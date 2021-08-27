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
$ markovian simple generate --encoding=string resources/Moby_Names_M_lc.txt --count=5
stergiramessey
barnan
bralph
jerrel
jord
```

If you only want names starting with "jo" you can use

```
$ markovian simple generate --encoding=string resources/Moby_Names_M_lc.txt --count=5 --prefix=jo
jone
jonathan
jon
joshuah
jostophanis
```

If you want them ending in "ton" then

```
$ markovian simple generate --encoding=string resources/Moby_Names_M_lc.txt --count=5 --suffix=ton
rounton
elston
janyatton
milton
brocharleton
```

You can even use both prefix and suffix at the same time, though quality seems to suffer a bit more

```
$ markovian simple generate --encoding=string resources/Moby_Names_M_lc.txt --count=5 --suffix=ton --prefix=jo
jonalluigiston
joakolarleton
joaquinton
josideighton
jonalluilton
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

You can add ` --prefix=prefix`, `--suffix=suffix` and `--count=N` to this too.

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


# Grammar based generation

Markovian has an aditional method for generating text - based on grammars.

To run it you use

```
markovian grammar --rules-file file.rul --start-token TOKEN
```

if `FantasyNovel.rul` contained

```
2 TITLE => "A " THING " of " ATTRIBUTE " and " ATTRIBUTE
1 TITLE => ATRIBUTE THING
1 THING => "Throne" | "Queen" | "King" | "Prince" | "Dance" | "House" | "Family"
1 ATTRIBUTE => "Fire" | "Ice" | "Death" | "Hope"
```

Then using `TITLE` as the token would general something like: "Fire Queen", "A
House of Fire and Fire" or "A Family of Death and Hope".

Note that the `--rules-file` is relative to the `--library-directory`, which
defaults to `.`.

This can be used for word generation - for example `mock_elvish.rul` uses a
structure based on triples of vowel-consonant-vowel or consonant-vowel-consonant
treating 'ay', 'ey' as vowels.

```
1 V_ => "a" | "e" | "i" |  "o" | "u"
1 V => V_ | V_ "y"
1 C => "r" | "n" | "f" | "k" | "l" | "c"
1 VTRIP => V_ C V
1 CTRIP => C V_ C
1 TRIP => VTRIP | CTRIP
1 ST => V_ C
1 A => TRIP TRIP
1 A => ST TRIP
1 A => CTRIP CTRIP V C " the " ST TRIP
```

This generates names like "cencorik the elerey", "feflar", "enfun". These are
admittedly pretty bad, but with some tuning of the language should be able to
produce something useful.

## Including other files into grammars

Lists of values can be imported as symbols, the example in `resources/fantasy_character/main.rul`

```
1 NAME => "The " PREFIX_ADJ " " TYPE " of the " SUFFIX
1 NAME => "A " TYPE " of the " PLACE
1 PLACE => PREFIX_ADJ " " SUFFIX
@import_list( "prefix_adj.txt" PREFIX_ADJ )
@import_list( "type.txt" TYPE )
@import_list( "suffix_noun.txt" SUFFIX )
```

```
> markovian grammar \
   --rules-file main.rul \
   --start-token NAME \
   --library-directory resources/rules/fantasy_character/
A Sorceress of the Black Shadows
...
The White Knight of the Desert
```

Other rule files can be imported too.

```
@import_language( "main.rul" )
1 ACTION => NAME " fought " NAME
1 ACTION => NAME " married " NAME
1 ACTION => NAME " ruled " PLACE

# A rule fil can have empty lines
# and lines starting with `#` are
# comments

# We can add to rules defined in other
# files
1 NAME => "John"
1 NAME => "Jane"
1 TYPE => "Clown"
```

This might generate "A Sorceress of the Black Shadows fought The White Knight of
the Desert", or "John married The Dark Clown of the East", or "Jane ruled Red
South". (Yep, you need to be pretty careful or you'll sometimes get nonsense)

# 3rd party components

Building an application often uses a lot of third-party parts.

A complete list with their license information can be found in [thirdparty.md](thirdparty.md).
