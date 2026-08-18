# Dedupit

## Project Description

This is a dataset "deduplicator" that I built from scratch. It is publicly hosted at www.dedupit.com.

Dedupit takes a CSV or TSV file and helps you remove duplicate rows in two stages: exact duplicates (rows that match perfectly on a set of columns you choose) are dropped automatically, while "fuzzy" duplicates (rows that don't match perfectly due to missing values) are grouped together and presented for manual review, group by group, before anything is deleted. The cleaned result can be saved to your computer at any point.

## Technologies Used

Python — core language
Flask — web framework and routing
pandas / NumPy — the actual deduplication logic (sorting, exact-match dropping, and a vectorized fuzzy-matching/clustering algorithm)
Jinja2 — server-rendered HTML templates
Vanilla JavaScript — dynamic form sections (variable numbers of sort columns / matching pairs) and the save/export flow, including the File System Access API (with a plain-download fallback for browsers that don't support it)
HTML / CSS — no frontend framework or build step
PythonAnywhere — hosting, running under uWSGI
GoDaddy DNS — custom domain routing for dedupit.com

## Features

Upload a CSV or TSV file and configure deduplication rules directly in the browser (no need to edit the file beforehand)
Exact Duplicate Detection — pick columns that must match perfectly; matching rows are dropped automatically, no review needed
Fuzzy Matching — pick "matching" columns (must match exactly) and "wildcard" columns (a blank value is allowed to match anything), to catch duplicates that are only incomplete, not identical
Data Completeness priority — optionally keep whichever version of a duplicate has the fewest blank fields, instead of just "whichever came first"
Sort Order configuration — control which row is treated as the "original" when duplicates are found
Group-by-group review UI (Delete All Duplicates / Delete Selected / Keep All) with color-coded rows so it's always clear what will be kept
A safety cap on pathologically large fuzzy-match groups (e.g. a column where thousands of unrelated rows share one value), with a clear on-page warning explaining what happened and why, instead of the app hanging or crashing
Save the cleaned file at any point, or at the end — the browser (not the server) decides where it's saved, using a native "choose a folder" dialog in supporting browsers
Cross-process-safe state: recovers in-progress work even if the app restarts or a request happens to land on a different backend worker

To-do List
- Create a more aesthetically pleasing UI
- Add option to consider strings that are similar (for example, "Jessica" and "Jesisca") as matching 

## Getting Started

Visit www.dedupit.com to use the deduplicator. Alternately, if you want to run the application locally, follow the steps below: 

1. Installing:
- Terminal commands:
  - git clone <your-repo-url>
  - cd Dedupit
  - pip install -r requirements.txt
2. Running: 
- start the application by using command "python app.py"
- open http://127.0.0.1:5000/ in your browser.

## Usage
1. Upload a CSV or TSV file on the home page.
2. Configure how duplicates should be detected:
3. Check columns under Exact Duplicate Detection for anything that should count as a duplicate only when it matches perfectly.
4. Add a Fuzzy Matching pair for anything that should count as a duplicate even with some missing information (choose the columns that must match exactly, and the columns where a blank is allowed to match anything).
5. Optionally enable Data Completeness to prefer keeping the more fully filled-in version of a duplicate.
6. Optionally set a Sort Order to control which row is treated as the original.
7. Click Process & Review Duplicates.
8. Step through each flagged group on the review page, choosing Delete All Duplicates, Delete Selected, or Keep All for each one.
9. Click Save current data frame at any point, or Save final data frame once you've been through every group, and choose where to save the cleaned CSV on your own computer.
