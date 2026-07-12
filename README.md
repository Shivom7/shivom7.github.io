# Shivom Gupta — Academic Portfolio

A responsive, accessible research portfolio for cognitive neuroscience, scientific computing and astrophysics. The site uses semantic HTML, modern CSS and minimal JavaScript, and deploys directly on GitHub Pages.

## Local preview

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. No build step or package installation is required.

## Structure

- `index.html` — main portfolio
- `eeg-pipeline.html` — EEG documentation scaffold
- `assets/css/portfolio.css` — design system and responsive layout
- `assets/js/portfolio.js` — mobile navigation and copyright year
- `images/` — portraits, project images and CV
- `sitemap.xml`, `robots.txt` — search engine discovery

## Deployment

Push the site to the default branch of the `Shivom7/Shivom7.github.io` repository. In **Settings → Pages**, select **Deploy from a branch**, then the repository root. GitHub Pages will publish it at `https://shivom7.github.io/`.

## Content updates

Replace the portrait placeholders on the exoplanet and Sakshi Sense project cards with representative images. Keep descriptive `alt` text, dimensions where known, and `loading="lazy"` for below-the-fold images. Complete the EEG documentation only after reviewing the actual source code and example outputs.
