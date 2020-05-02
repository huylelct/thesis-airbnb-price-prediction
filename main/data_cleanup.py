import pandas as pd
import math

# import data set
filename = '../data/listings.csv'
data = pd.read_csv(filename)

data = pd.DataFrame.drop(data, columns=[
    'listing_url',
    'scrape_id',
    'last_scraped',
    'name',
    'experiences_offered',
    'thumbnail_url',
    'medium_url',
    'picture_url',
    'xl_picture_url',
    'host_id',
    'host_url',
    'host_name',
    'host_location',
    'host_since',
    'host_about',
    'host_acceptance_rate',
    'host_thumbnail_url',
    'host_picture_url',
    'host_neighbourhood',
    'neighbourhood',
    'neighbourhood_cleansed',
    'neighbourhood_group_cleansed',
    'zipcode',
    'market',
    'country_code',
    'country',
    'minimum_minimum_nights',
    'maximum_minimum_nights',
    'minimum_maximum_nights',
    'maximum_maximum_nights',
    'minimum_nights_avg_ntm',
    'maximum_nights_avg_ntm',
    'calendar_updated',
    'calendar_last_scraped',
    'license',
    'jurisdiction_names',
    'summary',
    'space',
    'description',
    'neighborhood_overview',
    'notes',
    'transit',
    'access',
    'interaction',
    'house_rules',
    'street',
    'city',
    'state',
    'smart_location',
    'square_feet',
    'weekly_price',
    'monthly_price',
    'availability_30',
    'availability_60',
    'availability_90',
    'availability_365',
    'security_deposit',
    'cleaning_fee',
    'first_review',
    'last_review',
    # review process fields
    'amenities'
])


def is_nan(string):
    return string != string


def convert_host_resp_time(entry):
    array_host_resp_time = {
        'within an hour': 1,
        'within a few hours': 2,
        'within a day': 3,
        'a few days or more': 4,
    }
    if is_nan(entry):
        return 4
    return array_host_resp_time[entry]


data["host_response_time"] = data["host_response_time"].apply(convert_host_resp_time)


def remove_percent_sign(entry):
    if type(entry) == str:
        return entry.replace('%', '')
    else:
        return 0


data["host_response_rate"] = data["host_response_rate"].apply(remove_percent_sign)


def clean_boolean_value(entry):
    if entry == 't':
        return 1
    else:
        return 0


data["host_is_superhost"] = data["host_is_superhost"].apply(clean_boolean_value)
data["host_has_profile_pic"] = data["host_has_profile_pic"].apply(clean_boolean_value)
data["host_identity_verified"] = data["host_identity_verified"].apply(clean_boolean_value)
data["is_location_exact"] = data["is_location_exact"].apply(clean_boolean_value)
data["has_availability"] = data["has_availability"].apply(clean_boolean_value)
data["requires_license"] = data["requires_license"].apply(clean_boolean_value)
data["instant_bookable"] = data["instant_bookable"].apply(clean_boolean_value)
data["is_business_travel_ready"] = data["is_business_travel_ready"].apply(clean_boolean_value)
data["require_guest_profile_picture"] = data["require_guest_profile_picture"].apply(clean_boolean_value)
data["require_guest_phone_verification"] = data["require_guest_phone_verification"].apply(clean_boolean_value)

host_verification_set = set()


def collect_host_verifications(entry):
    entry_list = entry.replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace(" ", "").split(',')
    for verification in entry_list:
        if verification != "" and verification != 'None':
            host_verification_set.add(verification + "_verification")


data['host_verifications'].apply(collect_host_verifications)


def generic_verification(entry, v):
    entry_list = str(entry).replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace(" ", "").split(
        ',')
    for verification in entry_list:
        if verification + "_verification" == v:
            return 1
    return 0


for v in host_verification_set:
    data.insert(len(list(data)), v, 0)
    data[v] = data['host_verifications'].apply(lambda x: generic_verification(x, v))

data = pd.DataFrame.drop(data, columns=['host_verifications'])

property_type_set = set()


def collect_property_types(entry):
    if is_nan(entry):
        print("NAN")
    property_type_set.add(entry)


data["property_type"].apply(collect_property_types)


def encode_property_type(entry):
    return list(property_type_set).index(entry)


data["property_type"] = data["property_type"].apply(encode_property_type)

room_type_set = set()


def collect_room_types(entry):
    if is_nan(entry):
        print("NAN")
    room_type_set.add(entry)


data["room_type"].apply(collect_room_types)


def encode_room_type(entry):
    return list(room_type_set).index(entry)


data["room_type"] = data["room_type"].apply(encode_room_type)

bed_type_set = set()


def collect_bed_types(entry):
    if is_nan(entry):
        print("NAN")
    bed_type_set.add(entry)


data["bed_type"].apply(collect_bed_types)


def encode_bed_type(entry):
    return list(bed_type_set).index(entry)


data["bed_type"] = data["bed_type"].apply(encode_bed_type)

cancel_policy_set = set()


def collect_cancel_policy(entry):
    if is_nan(entry):
        print("NAN")
    cancel_policy_set.add(entry)


data["cancellation_policy"].apply(collect_cancel_policy)


def encode_cancel_policy(entry):
    return list(cancel_policy_set).index(entry)


data["cancellation_policy"] = data["cancellation_policy"].apply(encode_cancel_policy)


def clean_price(entry):
    if type(entry) != str and math.isnan(entry):
        return -55
    entry1 = entry.replace('$', '').replace(',', '')
    if float(entry1) == 0:
        return -55
    return float(entry1)


data["price"] = data["price"].apply(clean_price)
data["extra_people"] = data["extra_people"].apply(clean_price)


def fill_nan_with_zeros(entry):
    if is_nan(entry):
        return 0
    return entry


data["host_listings_count"] = data["host_listings_count"].apply(fill_nan_with_zeros)
data["host_total_listings_count"] = data["host_total_listings_count"].apply(fill_nan_with_zeros)
data["bathrooms"] = data["bathrooms"].apply(fill_nan_with_zeros)
data["bedrooms"] = data["bedrooms"].apply(fill_nan_with_zeros)
data["beds"] = data["beds"].apply(fill_nan_with_zeros)
data["reviews_per_month"] = data["reviews_per_month"].apply(fill_nan_with_zeros)


# review to choose suitable value
data["review_scores_rating"] = data["review_scores_rating"].apply(fill_nan_with_zeros)
data["review_scores_accuracy"] = data["review_scores_accuracy"].apply(fill_nan_with_zeros)
data["review_scores_cleanliness"] = data["review_scores_cleanliness"].apply(fill_nan_with_zeros)
data["review_scores_checkin"] = data["review_scores_checkin"].apply(fill_nan_with_zeros)
data["review_scores_communication"] = data["review_scores_communication"].apply(fill_nan_with_zeros)
data["review_scores_location"] = data["review_scores_location"].apply(fill_nan_with_zeros)
data["review_scores_value"] = data["review_scores_value"].apply(fill_nan_with_zeros)


print(len(data.columns))
print(data["host_listings_count"].describe())

data.to_csv("../data/listings_cleaned.csv")
