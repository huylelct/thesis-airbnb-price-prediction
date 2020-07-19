# Data desciption

### Data table
| Table | Description |
|---|---|
|Calendar| detailed calendar data for listings, including listing id and the price and availability for that day|
|Listing| detailed listings data including full descriptions and average review score |
|Reviews| detailed review data for listings including unique id for each reviewer and detailed comments|
|Listing summary| summary information and metrics for listings (good for visualisations) |
|Review summary| summary Review data and Listing ID (to facilitate time based analytics and visualisations linked to a listing)|
|Neighborhood| neighborhood list for geo filter. Sourced from city or open source GIS files|

##### Calendar table
Calendar table desciption
|Column|Type|Desciption|
|---|---|---|
|listing_id|integer||
|date|date||
|available|boolean|nhà có sẵn hay không|
|price|float|giá|
|adjusted_price|float|giá đã điều chỉnh|
|minimum_nights|integer|số đêm thuê tối thiểu|
|maximun_nights|integer|số đêm thuê tối đa|

##### Listings table
Listings table description
|Coulumn|Type|Description|
|---|---|---|
|id|integer||
|listing_url|string|url danh sách phòng trên airbnb (ex: https://www.airbnb.com/rooms/17878	)
|scrape_id||
|last_scraped|date|lần cuối cào dữ liệu|
|name|string|tên listing (Ex: Beautiful Modern Decorated Studio in Copa)
|summary|string|tóm tắt về phòng|
|space|string|giới thiệu về không gian  phòng|
|description|string| mô tả về phòng|
|experiences_offered|string| kinh nghiệm được cung cấp (value: none -100%)
|neighborhood_overview|string|mô tả về những người hàng xóm xung quanh|
|notes|string| lưu ý về căn phòng |
|transit|string| phương tiện đi lại công cộng (taxi, bus , ...)|
|access|string| thông tin các tiện ích có thể sử dụng|
|interaction|string| khả năng tương tác với host khi cần thiết|
|house_rules|string| quy định sử dụng nhà |
|thumbnail_urls||null - 100%|
|medium_urls||null - 100%|
|picture_url|string|url các ảnh nhà|
|xl_picture_url||null-100%|
|host_id|||
|host_url|string|url profile host|
|host_name||
|host_since||
|host_location||
|host_about||
|host_response_time|string| thời gian host phản hồi|
|host_response_rate|string| tỉ lệ thắc mắc host phản  hồi|
|host_acceptance_rate||N/A(100%)
|host_is_superhost|bool|
|host_thumbnail_url||
|host_picture_url||
|host_neighborhood|string|khu vực lân cận host|
|host_listings_count|integer| số lượng listing của host
|host_total_listings_count|integer| tổng số lượng listing của host
|host_verifications|array|mảng các phần mà host đã đc verified
|host_has_profile_pic|boolean|host có avt hay không
|host_identify_verified|boolean| host đã đc verified hay chưa
|street||
|neighbourhood|string| vùng lân cận
|neighborhood_cleansed||
|neighbourhood_group_cleansed|| null-100%
|city||
|state||
|zipcode||
|market||
|smart_location||
|country_code||
|country||
|latitude|float| vĩ độ
|longitude|float| kinh độ
|is_location_exact|boolean|vị trí chính xác cụ thể hay không
|property_type|string|loại nhà (apartment, house,...)
|room_type|string| loại phòng (Entire home/apt, private room ,...)
|accommodates|integer| sức chứa
|bathrooms|float| số lượng phòng tắm
|bedrooms|integer| số lượng phòng ngủ
|beds|integer| số lượng giườn
|bed_type|string| loại giường
|amenities|json| tiện nghi
|square_feet|integer|diện tích|
|price|float|giá|
|weekly_price||
|monthly_price||
|security_deposit|float| tiền cọc
|cleaning_fee|float|phí vệ sinh|
|guest_included|integer| có thể bao gồm bao nhiều khách
|extra_people|float| phí cho thêm người
|minimum_nights||
|maximum_nights||
|minimum_minimum_nights||
|maximun_minimum_nights||
|minimum_maximum_nights||
|maximum_maximum_nights||
|minimum_nights_avg_ntm||
|maximum_nights_avg_ntm||
|calendar_updated||
|has_availability|boolean|có sẵn hay không
|availability_30|integer| có sẵn bao nhiêu ngày trong 30 ngày tới
|availability_60|integer| có sẵn bao nhiêu ngày trong 60 ngày tới
|availability_90|integer| có sẵn bao nhiêu ngày trong 90 ngày tới
|availability_365|integer| có sẵn bao nhiêu ngày trong 365 ngày tới
|calendar_last_craped||
|number_of_reviews||
|number_of_reviews_ltm||
|first_review||
|last_review||
|review_scores_ratings||
|review_scores_accuracy||
|review_scores_cleanliness||
|review_scores_checkin||
|review_scores_communication||
|review_scores_location||
|review_scores_value||
|require_license||
|license||null-100%
|jurisdiction_names||null-100%
|instant_bookable|bool| có thể đặt ngay hay không
|is_business_travel_ready|bool|
|cancellation_policy|string|chính sách hoãn (flexible, strict)
|require_guest_profile_picture|boolean| 
