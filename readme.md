# Predicting Box Office Sales

## Description:
This is a relatively simple project to determine whether features such as a movie's budget, score in online reviews (both by professional critics and the general public), number of online reviews, rating (G -> PG -> PG-13 -> R), and runtime can be used to predict the movie's box office sales.

## Objective:
To determine whether or not the aforementioned features can be used to predict box office sales, and if so, to what extent?

## Methodology:
I used Selenium and BeautifulSoup to scrape data from BoxOfficeMojo and Metacritic. BoxOfficeMojo was straightforward to scrape, while Metacritic was not.

All the features I began with:
- budget - how much was spent in the making of the movie
- metascores - according to Metacritic, the "world's most respected critics" write reviews on the site, these reviews are assigned a score by Metacritic, then a weighted average is applied to summarize the range of their opinions - this is the metascore, a number from 0-100
- number of critics - the number of critics who contributed reviews from which the movie's metascore is generated
- audience score - a score between 0-10 (with one decimal place) generated by collecting ratings from anyone who wants to rate the movie
- number of audience ratings - the number of people who contributed an audience rating
- runtime - length of the movie in minutes
- domestic total gross - the target variable - the amount of money the movie made in sales at the box office
- rating - the rating of the movie, ranging from G -> PG -> M/PG -> PG-13 -> R. To a small extent, limits the amount of people who can see a movie (only R ratings - children need parental supervision to attend an R rated movie)

I used LASSO regularization to determine which features I should keep in the model, and which I should discard. The important features ended up being budget and number of audience reviews (somewhat surprisingly to me).

## Results: <br>
The first model I tried was simple linear regression, statsmodel OLS with all of the features listed above included. The R^2 was .548 adjusted. The condition number for this first test was not good - 1.25 x 10^17.

Next, I ran the training set through LASSO with increasingly high alpha values until I was left with only two non-zero coefficients - for the features budget and number of audience ratings. When I tried plugging only these two features into statsmodel OLS, the R^2 went up to .655 adjusted, and the condition number went down to 1.63. Depending on how important the condition number is, this may be overall my best result. I didn't know that at the time though, and I kept trying other approaches.

Next I tried Polynomial Regression, which produced the second best R^2 result - 0.712 on the test data. In this model I was only using two features - budget and an interaction term consisting of number of audience ratings * number of critic ratings. The next model I tried was linear regression with LassoCV. I used two features again, but instead of the index_count this time, I went back to the simpler budget and number of audience ratings. I got an adjusted R^2 of .734, but the condition number was still bad - 1.08 x 10^5. Based on how close these results were to the polynomial results, I have come to the conclusion that my model lies somewhere between linear and polynomial, but since the linear model produced a slightly higher R^2, I'll go with more linear than polynomial.

I was able to disprove my null hypothesis, that the number of audience ratings is completely unrelated to box office sales. Next steps would include more rigorous cross validation testing, and more/better data.