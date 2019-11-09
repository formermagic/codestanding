# Project Proposal

## Plan

1. Read articles and write brief descriptions of thesis ideas
2. Collect available datasets from read articles
3. Describe experiments I want to reproduce
4. Develop a tool that collects and prepares data from public repositories (diffs, code bases, etc)
5. Prepare a basic baseline (i.e. reproduce an experiment)

Note, that by step 4 I should have a certain understanding of data (e.g. type, format, mining technique, etc).

## Data Format

* **repo** name
* **url** url
* **sha** sha
* **language** language
* **message** commit.message
* **files** files
    * **sha** sha
    * **name** filename % name
    * **extension** filename % ext
    * **blob_url** blob_url
    * **raw_url** raw_url
    * **patch** patch
    * **patch_tokens** -empty-
    * **partition** -empty-
    * **status** status
    * **changes** changes
    * **additions** additions
    * **deletions** deletions
* **stats** stats
    * **changes** total
    * **additions** additions
    * **deletions** deletions