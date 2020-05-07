import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def draw_bar_chart(data_chart, title):
    data_bar_chart = {}
    for item in data_chart:
        if item in data_bar_chart:
            data_bar_chart[item] += 1
        else:
            data_bar_chart[item] = 1

    np_array = np.array(list(data_bar_chart.items()))
    # if title == "host_listings_counts":
    #     print(np_array)
    labels = np_array[0:, 0]
    values = np_array[0:, 1]

    fig, ax = plt.subplots()

    ax.bar(labels, values, 0.35)
    ax.set_title(title)
    plt.savefig("../data_visualization/" + title + ".png")


if __name__ == "__main__":
    data = pd.read_csv("../data/listings_cleaned.csv")
    for column in data.columns:
        draw_bar_chart(data[column], column)
    # draw_bar_chart(data["host_response_time"], "host_response_time")
    # draw_bar_chart(data["host_is_superhost"], "host_is_superhost")
    # draw_bar_chart(data["host_listings_count"], "host_listings_counts")
    # draw_bar_chart(data["host_total_listings_count"], "host_total_listings_count")
    # draw_bar_chart(data["host_has_profile_pic"], "host_has_profile_pic")
    # draw_bar_chart(data["host_identity_verified"], "host_identity_verified")
    # draw_bar_chart(data["is_location_exact"], "is_location_exact")
    # draw_bar_chart(data["property_type"], "property_type")
    # draw_bar_chart(data["room_type"], "room_type")
    # draw_bar_chart(data["accommodates"], "accommodates")
    # draw_bar_chart(data["bathrooms"], "bathrooms")
    # draw_bar_chart(data["bedrooms"], "bedrooms")
    # draw_bar_chart(data["beds"], "beds")
    # draw_bar_chart(data["bed_type"], "bed_type")
    # draw_bar_chart(data["guests_included"], "guests_included")
    # draw_bar_chart(data["extra_people"], "extra_people")
    # draw_bar_chart(data["minimum_nights"], "minimum_nights")
    # draw_bar_chart(data["maximum_nights"], "maximum_nights")
    # draw_bar_chart(data["has_availability"], "has_availability")
    # draw_bar_chart(data["number_of_reviews"], "number_of_reviews")
    # draw_bar_chart(data["number_of_reviews_ltm"], "number_of_reviews_ltm")
    # draw_bar_chart(data["review_scores_rating"], "review_scores_rating")
    # draw_bar_chart(data["review_scores_accuracy"], "review_scores_accuracy")
    # draw_bar_chart(data["review_scores_cleanliness"], "review_scores_cleanliness")
