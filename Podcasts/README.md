<h1>Podcasts-downloader.ipynb: Programatic scrape, save to AWS S3 and Transcribe podcast</h1>

<p><b>You need to fill aws_credentials.json file with your AWS creadentials</b></p>

<h2>Steps:</h2>
<ul>
    <li><b>Web scraping:</b> Scrape podcast items by given url from https://feeds.buzzsprout.com/</li>
    <li>Extract URLs to .mp3 files for the podcasts</li>
    <li>Download all .mp3 files to /downloads folder</li>
    <li>Upload files to AWS S3 buckets</li>
    <li>Trascribe podcasts using AWS Transcribe</li>
    <li>Save results to downloads folder</li>
</ul>

<h1>Trascribe-AssemblyAI.ipynb: Upload and Transcribe podcast using <b>Assembly AI</b></h1>

<p><b>You need to fill assembly_ai_credentials.json file with your AssemblyAI creadentials</b></p>
<p>For more info: https://www.assemblyai.com/app/</p>