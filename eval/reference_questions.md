# Reference questions — Phase 1 benchmark

Purpose: score retrieval quality, cross-lingua behaviour, persona, retcon
handling, and (later) the local-vs-API comparison — on the *same* questions
every time. Score with `SCORING.md`. Don't cherry-pick: run all 30.

Legend of what each question stresses:
[L] direct lookup · [X] cross-source synthesis · [IT] Italian query on the
English corpus · [C] contradiction / unreliable narrator · [F] flavour-text
retrieval (proper-noun heavy → hybrid search test) · [N] negative control
(the honest-Ghost test) · [S] spoiler flag (Phase 2 warning test)

## A. Direct lookups — retrieval sanity
1.  [L] What is the Traveler?
2.  [L] Who is Cayde-6 and how did he die?
3.  [L] What are Ghosts and where do they come from?
4.  [L] What is the Last City?
5.  [L] Who are the Nine?
    *Expect: honest handling — even in-universe the answer is elusive; a
    good response says what is attested and flags the ambiguity as canon.*
6.  [L][IT] Chi è Zavala e qual è il suo ruolo nell'Avanguardia?
7.  [L] What is the Darkness, and how did its definition evolve across
    releases? *(also a soft [C]: early vagueness vs later Witness canon)*

## B. Proper nouns & flavour text — the hybrid-search stress test
8.  [F] What does "aiat" mean and where does it appear?
9.  [F] What is the significance of the phrase "o bearer mine"?
10. [F] Which weapons carry flavour text written by or about the Drifter?
11. [F][IT] Cosa sono gli Ahamkara e perché i loro ossi sussurrano?
12. [F] What does the flavour text of Ace of Spades tell about Cayde?

## C. Cross-source synthesis — the reason this project exists
13. [X] Reconstruct the relationship between Savathûn and Osiris across
    releases, from her earliest mentions to the Witch Queen era.
14. [X] Trace Uldren Sov's arc: prince, murderer, Crow. Which sources
    cover each stage?
15. [X] How do the Books of Sorrow, later lore, and Ahsa's revelations
    each describe the Hive's pact with the Worm Gods? What changed?
    *(also [C]: the Books of Sorrow are propaganda — attribution required)*
16. [X] What do we know about Clovis Bray (the man), assembling corporate
    records, Europa lore and Exo history?
17. [X][IT] Ricostruisci la storia di Eris Morn: dalla Cripta di Crota fino
    al suo ruolo contro la Strega. Da quali libri proviene ogni fase?
18. [X] What is the Deep/Sword logic vs the Sky/Garden imagery? Collect
    the philosophy scattered across grimoire-adjacent and D2 sources.
19. [X] Map Mara Sov's plans-within-plans around the Dreaming City
    curse: what did she know and when?

## D. Contradictions, retcons, unreliable narrators
20. [C] Who created the Vex? Present every account the archive holds and
    date them. *(classic ambiguity; a model that gives one clean answer fails)*
21. [C] Was Rezyl Azzir's fall to become Dredgen Yor inevitable? Contrast
    the tellings.
22. [C] The Great Disaster at Twilight Gap vs the Battle of Six Fronts —
    do sources ever blur them? Attribute carefully.
    *(also [N]-ish: if the corpus is thin here, saying so beats inventing)*
23. [C][IT] Il Viaggiatore è fuggito dal Crogiolo della guerra tra Eliksni
    e l'umanità o è rimasto per scelta? Confronta le versioni degli Eliksni
    ("il Grande Distacco") con le fonti dell'Ultima Città.

## E. Negative controls — the honest-Ghost test
24. [N] What did Master Rahool do during the Red War? *(likely thin/absent:
    the correct answer admits the archive says little, without inventing)*
25. [N] Which lore book describes the childhood of Lord Shaxx? *(trick:
    almost certainly none — the Ghost must say so)*
26. [N][IT] Esiste una voce di lore che spiega perché i Cabal hanno paura
    dei Psion? *(premise is wrong/reversed — must push back, not confirm)*

## F. Spoiler-flag behaviour (records Phase 2, warning testable in Phase 1)
27. [S] What was Osiris doing during Season of the Lost?
    *(chunks flagged spoils=[witch_queen] → expect the pre-spoiler warning)*
28. [S] What did Mara Sov's coven do with the Techeuns before The Witch
    Queen launched? *(same flag; after user says "continue", no re-warning)*

## G. Persona & citation-on-request
29. Tell me the story of the Dreaming City curse as if we were flying
    there right now. *(pure persona test: storytelling, no citations,
    no "according to the sources")*
30. Follow-up to 29: "Where exactly does all that come from?"
    *(citation-on-request test: expect precise book/entry names only now)*
