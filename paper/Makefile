# =============================================================================
# @file    Makefile
# @brief   Makefile for generating previews of the paper
# @author  Michael Hucka <mhucka@caltech.edu>
# @license Please see the file named LICENSE in the project directory
# @website https://github.com/casics/dassie
# =============================================================================

# Change the following values to match your configuration.
# .............................................................................

assets   := assets
input    := paper.md
output   := paper.pdf
bib      := paper.bib
bibstyle := apa.csl

# Main code -- no more customization variables after this point
# .............................................................................

title	:= $(shell grep title: $(input) | sed 's/title: *//' | tr -d "'")
authors := $(shell sed -n '/authors:/,/affiliations:/p' $(input) | grep name: | sed 's/- name: *//' | paste -d, -s - | sed 's/,/, /g')
repo	:= $(shell git remote get-url origin | sed 's|git@github.com:|https://github.com/|' | sed 's/\.git//')

$(output): $(input) $(bib) Makefile
	pandoc \
	--from markdown+autolink_bare_uris+implicit_figures \
	-V paper_title="$(title)" \
	-V citation_author="$(authors)" \
	-V repository="$(repo)" \
	-V archive_doi="http://dx.doi.org/00.00000/zenodo.0000000" \
	-V formatted_doi="00.00000/joss.00000" \
	-V paper_url="http://joss.theoj.org/papers/" \
	-V review_issue_url="http://joss.theoj.org/papers/" \
	-V issue="0" \
	-V volume="00" \
	-V year="2020" \
	-V submitted="25 February 2020" \
	-V published="25 February 2020" \
	-V page="00" \
	-V graphics="true" \
	-V geometry:margin=1in \
	-V logo_path=$(assets)/logo.png \
	--filter pandoc-citeproc --csl $(assets)/$(bibstyle) \
	--template $(assets)/latex.template \
	$< -o $@ \

clean:
	rm -rf *.pdf
