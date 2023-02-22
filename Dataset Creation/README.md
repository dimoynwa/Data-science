<h1>Programatic scrape Wikipedia Disney page</h1>

<p><b>You need to fill omdb_credentials.json file with your OMDB creadentials.</b> You can create a free account at 'http://www.omdbapi.com/'</p>

<h2>Steps:</h2>
<ul>
    <li><b>Web scraping:</b> Scrape Wikipedia page: https://en.wikipedia.org/wiki/List_of_Walt_Disney_Pictures_films</li>
    <li>Find all tables of class <b>wikitable</b></li>
    <li>Loop tables and extract URL for all the movies</li>
    <li>Scrape all movies pages and create a movie-info</li>
    <li>Clean up: parse budgets and box-offices as numbers, realease date as datetime objects</li>
    <li>Call OMDB API to extract imbd, Rotten tommatoes ratings for all the movies and result to dataset</li>
    <li>Save result as JSON and CSV in movies folder</li>
</ul>