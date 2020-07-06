import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


def draw_distribute_plot(data_x, bins, filename, title, x_label='', y_label=''):
    # print(data_x)
    fig, ax = plt.subplots()
    data_x = data_x.to_numpy()

    if bins > 0:
        ax.hist(data_x, bins)
    else:
        ax.hist(data_x)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    plt.savefig("../data_visualization/" + filename + ".png")


def filter_array(list_data, max_element):
    filter_array = []
    fail_value = 0
    for element in list_data:
        if element > max_element:
            fail_value += 1
            filter_array.append(False)
        else:
            filter_array.append(True)
    return list_data[filter_array], fail_value, max_element


def remove_zero_value(list_data):
    filter_array = []
    total = 0
    for element in list_data:
        if element == 0:
            filter_array.append(False)
        else:
            total += 1
            filter_array.append(True)
    return list_data[filter_array], total


def func(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)


def draw_type_home_chart():
    data_chart = {
        "Entire home/apt": 0,
        "Private room": 0,
        "Hotel room": 0,
        "Shared room": 0,
    }
    for type_home in original_data["type"]:
        data_chart[type_home] += 1

    labels = list(data_chart.keys())
    sizes = list(data_chart.values())

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')

    ax.set_title("Room type distribution (total: " + str(len(original_data)) + ")")

    plt.savefig("../data_visualization/type_room.png")


def draw_pie_chart(list_data, filename, title):
    data_chart = {
        "Yes": 0,
        "No": 0
    }

    for item in list_data:
        if item == 0:
            data_chart["No"] += 1
        else:
            data_chart["Yes"] += 1

    labels = list(data_chart.keys())
    sizes = list(data_chart.values())

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')

    ax.set_title(title)

    plt.savefig("../data_visualization/" + filename + ".png")


if __name__ == "__main__":
    data = pd.read_csv("../data/data_cleaned.csv")
    original_data = pd.read_csv("../data/airbnb-hcm.csv")

    draw_type_home_chart()
    draw_pie_chart(data["is_host_verified"], "host_verified", "Host verified distribution")
    draw_pie_chart(data["is_superhost"], "is_superhost", "Superhost distribution")
    draw_pie_chart(data["no_pets"], "no_pest", "No pets distribution")
    draw_pie_chart(data["can_parties"], "can_party", "Can party distribution")
    draw_pie_chart(data["can_smoking"], "can_smoking", "Can smoking distribution")

    draw_distribute_plot(data["price"], 25, "price", "Price distribution", "Price", "Count")
    draw_distribute_plot(data["guest"], 16, "guest", "Guest distribution", "Guest", "Count")

    data_chart, fail_value, max_element = filter_array(data["bedroom"], 6)
    title = "Bedroom distribution (>" + str(max_element) + ": " + str(fail_value) + ")"
    draw_distribute_plot(data_chart, 7, "bedroom", title, "Bedroom", "Count")

    data_chart, fail_value, max_element = filter_array(data["bed"], 6)
    title = "Bed distribution (>" + str(max_element) + ": " + str(fail_value) + ")"
    draw_distribute_plot(data_chart, 7, "bed", title, "Bed", "Count")

    data_chart, fail_value, max_element = filter_array(data["bath"], 6)
    title = "Bath distribution (>" + str(max_element) + ": " + str(fail_value) + ")"
    draw_distribute_plot(data_chart, 7, "bath", title, "Bath", "Count")

    data_chart, fail_value, max_element = filter_array(data["service_fee"], 20)
    title = "Service fee distribution (>" + str(max_element) + ": " + str(fail_value) + ")"
    draw_distribute_plot(data_chart, 20, "service_fee", title, "Service fee", "Count")

    data_chart, total_item = remove_zero_value(data["score"])
    title = "Score distribution (total: " + str(total_item) + ")"
    draw_distribute_plot(data_chart, 40, "score", title, "Score", "Count")

    data_chart, _, _ = filter_array(data["cleaning_fee"], 30)
    data_chart, total_item = remove_zero_value(data_chart)
    title = "Cleaning fee distribution (total: " + str(total_item) + ")"
    draw_distribute_plot(data_chart, 20, "cleaning_fee", title, "Cleaning fee", "Count")

    draw_distribute_plot(data["host_response_rate"], 50, "host_response_rate", "host_response_rate distribution",
                         "Host response rate", "Count")

    data_chart, fail_value, max_element = filter_array(data["reviews"], 80)
    title = "Reviews distribution (>" + str(max_element) + ": " + str(fail_value) + ")"
    draw_distribute_plot(data_chart, 30, "reviews", title, "Reviews", "Count")

    data_chart, fail_value, max_element = filter_array(data["host_reviews"], 1000)
    title = "Host reviews distribution (>" + str(max_element) + ": " + str(fail_value) + ")"
    draw_distribute_plot(data_chart, 30, "host_reviews", title, "Host reviews", "Count")
