# docs/source/conf.py

import os
import sys
import sphinx_rtd_theme

# -- Configuration de base --
project = 'Sleep Disorder Predictor'
copyright = '2025, Amri Maryam'
author = 'Amri Maryam'
release = '1.0.0'

# -- Extensions --
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
]

# -- Thème RTD --
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# -- Options du thème --
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_position': 'both',
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- Chemins statiques --
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['custom.js']

# -- Logo --
html_logo = '_static/images/logo.png'
#html_favicon = '_static/images/favicon.ico'

# -- Sidebar --
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}