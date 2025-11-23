# Data Preprocessing changes: Max
Instead of writing the full hotel name and user name I made a unqiue mapping to both and saved it. We can just use Integers afterall (memory efficeint)

I also made preprocessing always output a simple CSV file of user_id, item_id, rating for the simple recSys, it then also output the sharded more complex functions - [user_id, item_id, rating, [subratings]] can be written to a single csv ~1GB

# Setup
Download HotelRec dataset from [https://github.com/Diego999/HotelRec/tree/master] and run:
```bash
(create file with ratings and text - 30 shards)
python data/hotelrec_loader.py \
  --in /path/to/dataset/HotelRec.txt \
  --out-dir data/filtered \
  --shard-size 1000000 \
  --drop-prop

(create file with ratings and subrating - single shard) - run on 10 +
python data/hotelrec_loader.py \
  --in ./HotelRec.txt \
  --out-dir data/filtered \
  --shard-size 30000000 \
  --min-user-count 10 \
  --min-item-count 20 \
  --drop-text

python data/hotelrec_loader.py \
  --in ./HotelRec.txt \
  --out-dir data/filtered \
  --shard-size 30000000 \
  --drop-text
```

# Sample Data
AS you can see below we have access to a range of features; from basic rating to subratings and even textual reviews. 
```bash
[
  {
   "hotel_url":"Hotel_Review-g194775-d1121769-Reviews-Hotel_Baltic-Giulianova_Province_of_Teramo_Abruzzo.html",
   "author":"ladispoli",
   "date":"2010-02-01T00:00:00",
   "rating":4.0,
   "title":"Great customer service",
   "text":"Great customer service and good restaurant service is what made this experience so wonderful for my family. Giulianuova is a pretty simple , not so \"wow\" town, with not very clean beaches as I was expecting when booking my reservation. What saved my vacation was staying at Baltic. Baltic is a very simple but functional hotel, but what makes it so special are the people that work there. It seems that as a part of their job training they had take some kind of class on humanity and spirituality, because the way they treat every single person with respect and smile is just amazing. Good job to the manager Massimo (who knows how to hire great people)!",
   "property_dict":{
      "sleep quality":4.0,
      "value":4.0,
      "rooms":3.0,
      "service":5.0,
      "cleanliness":3.0,
      "location":3.0
      }
   }
]
```
# Preprocessing Description
The 50 million (~50GB) sparse review dataset poses a major memory challange. In order to determine which algorithm to use for recommendations, we first need to perform sparsity analysis and understand the cold start limitions of the data. As a result we learned that ~67% of users only made 1 review. For this reason we decided to limit the data to only the reviews of users with more 3 or more records. While resulting in a ~80% decrease in number of users, it only translates to a ~60% reduction in reviews. That way we still have ~30 million high quality interconnected reviews remaining, allowing us to compare similarity of behaviour between users and items while not worriying about the cold-start problem.


# Preprocessing Results
### Basic Analysis and filtering
```
✅ Phase 1 complete.
  scanned 50,264,531 | qualifying 50,264,531 reviews
  Total unique hotels: 365,057, authors: 21,891,404
  Users with ≥3 reviews: 4,081,761
  Items (Hotels) with ≥3 reviews: 365,007

  Scanned: 50,264,531  |  Kept: 29,432,641 reviews
```

### Sparsity Analysis
```
Sampled: 1,000,000 => Unique-Users: 755,340  Unique-Items: 3,765  Unique-Pairs: 914,406
Matrix Density: 0.00032154  Matrix Sparsity: 0.99967846
User p50/p90/p99: 1/2/4
Item p50/p90/p99: 179/543/1685

Average reviews per user: 2.240
Average reviews per Item: 134.357

===== USER ACTIVITY BUCKETS =====
=1  : 14,787,425 users  (67.55%)
=2  :  3,022,218 users  (13.81%)
=3  :  4,081,758 users  (18.65%)
≥5  :  2,030,351 users  (9.27%)
≥10 :    699,514 users  (3.20%)

SUMMARY
Pass1 qualifying rows: 50,264,531
Rows belonging to users <3: ~20,831,861 (overlaps with item drops)
Rows belonging to items <3: ~56 (overlaps with user drops)
Total scanned (pass2): 50,264,531
Total kept (written) : 29,432,641
Shards written       : 59

13 MILION REVIEWS FOR 10+ USER/ITEM RATINGS
```
