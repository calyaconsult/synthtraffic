## What is this?

**synthtraffic** is a Python tool that generates fake (but realistic) website traffic data. It imitates the visitors of an imaginary travel agency's website that sells tours to places like Oman, Dubai, Egypt, Prague and Vienna. Because the data is invented by rules and not collected from real people, it contains no personal information.

## What it produces

Each run creates a set of files that look like what a real web analytics system would record:

- **Sessions:** one row per website visit (where the visitor came from, device, country, how long they stayed, whether they booked)
- **Events:** the individual page views and booking steps within each visit
- **Daily KPIs:** summary figures per day, such as visitors, bounce rate, bookings, revenue and conversion rate
- **An access log:** in the style of a web server (Nginx) log
- **An answer key:** a file that says which sessions were bots and which days were unusual

With the default settings you get about 91 days of data: roughly 56,000 visits, 115,000 events and 1,100 bookings.

## Why the data is realistic

The simulation includes patterns you would see in real life: more visitors on Sundays, an evening peak around 8 pm, returning customers, seasonal ups and downs, and a booking process with drop-off at each step (request, details, confirmation). It also adds deliberate mess, such as unflagged bots, duplicate sessions, tracking errors, missing countries, and special days like a newsletter blast, a site outage or a TV feature.

## What it is used for

The README suggests using it to:

- test analytics pipelines and dashboards without touching real user data
- practise or evaluate bot and anomaly detection, since the answer key lets you score how well a method performs
- try out attribution models (which marketing channel gets credit for a booking)
- create SQL exercises and training material

## Practical notes

- It runs from the command line (`synthtraffic`) or from Python code.
- Settings such as the number of days, traffic volume, bot share and the random seed can be changed. The same seed gives the same data every time.
- You can also define your own website structure and event calendar.
- It includes 26 automatic checks that confirm the generated data is consistent and statistically plausible.
- The README is open about its limits: the numbers are plausible assumptions, not based on real data, so they are illustrative rather than accurate.
- It is released under the MIT license, so it can be freely used and shared.

**In one sentence:** it is a safe, repeatable "practice dataset factory" for anyone who wants to work with website analytics data without needing real, privacy-sensitive visitor data.
