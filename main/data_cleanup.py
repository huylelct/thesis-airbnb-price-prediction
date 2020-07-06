import pandas as pd
import math

# import data set
filename = '../data/airbnb-hcm-1.csv'
data = pd.read_csv(filename)

data = pd.DataFrame.drop(data, columns=[
    'main_image',
    'name',
    'amentity_unavailable:_carbon_monoxide_alarm\ncarbon_monoxide_alarm',
    'amentity_unavailable:_smoke_alarm\nsmoke_alarm',
    'id',
    'longitude',
    'latitude',
    # 'address',
    # 'type',
    # 'host_response_time',
    'half-bath'
])


def is_nan(string):
    return string != string


address_set = set()


def collect_address(entry):
    if entry in address_set:
        return
    address_set.add(entry)


data["address"].apply(collect_address)


def convert_address(entry, address):
    if entry in address:
        return 1
    return 0


data["is_dong_nai"] = data["address"].apply(
    lambda x: convert_address(x,
                              ['P. Hiệp Thành', 'Ho Nai', 'P. Tam Hòa', 'Tt. Long Thành', 'P. Tân Tiến', 'P. Tân Hiệp',
                               'P. Thống Nhất', 'P. Tam Hiệp', 'P. Tân Hòa', 'P. Tân Biên', 'P. Quyết Thắng',
                               'Xã Bắc Sơn', 'P. Hòa Bình', 'P. Tân Mai', 'Xã Trị An', 'Xã Lộc An']))
data["is_binh_duong"] = data["address"].apply(
    lambda x: convert_address(x,
                              ['P. Vĩnh Phú', 'P. Lái Thiêu', 'P. Chánh Nghĩa', 'P. Phú Hòa', 'P. Chánh Phú Hòa',
                               'Xã Hiếu Liêm', 'P. Tân Đông Hiệp', 'P. Hòa Phú', 'P. An Phú', 'P. Phú Tân',
                               'P. Thới Hòa', 'P. Đông Hòa', 'P. Dĩ An', 'P. Phú Thọ']))
data["is_ngoai_tinh"] = data["address"].apply(
    lambda x: convert_address(x, ['Xã Thạnh Lợi', 'Xã Bình Ân', 'Xã Tân Bình Thạnh', 'Tt. Cần Đước', 'Xã Bình Tâm',
                                  'Xã Kiểng Phước', 'Tiền Giang', 'Xã Phú Ngãi Trị', 'Xã Tân Mỹ', 'Xã Phú Kiết',
                                  'Xã Đạo Thạnh', 'Tt. Bến Lức', 'Xã Long Hựu Đông']))
data["is_phu_nhuan"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận Phú Nhuận']))
data["is_tan_phu"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận Tân Phú']))
data["is_binh_tan"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận Bình Tân']))
data["is_quan_4"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận 4']))
data["is_can_gio"] = data["address"].apply(
    lambda x: convert_address(x, ['H. Cần Giờ']))
data["is_nha_be"] = data["address"].apply(
    lambda x: convert_address(x, ['H. Nhà Bè']))
data["is_quan_5"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận 5', 'Phường 8']))
data["is_go_vap"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận Gò Vấp']))
data["is_binh_chanh"] = data["address"].apply(
    lambda x: convert_address(x, ['H. Bình Chánh']))
data["is_quan_12"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận 12']))
data["is_tan_binh"] = data["address"].apply(
    lambda x: convert_address(x, ['Tan Binh']))
data["is_quan_3"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận 3']))
data["is_quan_7"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận 7']))
data["is_hoc_mon"] = data["address"].apply(
    lambda x: convert_address(x, ['H. Hóc Môn']))
data["is_quan_9"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận 9']))
data["is_quan_8"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận 8']))
data["is_quan_6"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận 6']))
data["is_quan_2"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận 2']))
data["is_quan_10"] = data["address"].apply(
    lambda x: convert_address(x, ['Phường 1', 'Quận 10', 'Phường 7', 'Phường 2', 'Phường 4']))
data["is_quan_1"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận 1']))
data["is_quan_11"] = data["address"].apply(
    lambda x: convert_address(x, ['Quận 11']))
data["is_cu_chi"] = data["address"].apply(
    lambda x: convert_address(x, ['H. Củ Chi']))
data["is_thu_duc"] = data["address"].apply(
    lambda x: convert_address(x, ['Thu Duc']))
data["is_binh_thanh"] = data["address"].apply(
    lambda x: convert_address(x, ['Bình Thạnh']))

data = data.drop(columns=["address"])
print(address_set)


def check_type_room(entry, type_room):
    if entry == type_room:
        return 1
    return 0


data["is_entire_home"] = data["type"].apply(lambda x: check_type_room(x, "Entire home/apt"))
data["is_private_room"] = data["type"].apply(lambda x: check_type_room(x, "Private room"))
data["is_hotel_room"] = data["type"].apply(lambda x: check_type_room(x, "Hotel room"))
data["is_shared_room"] = data["type"].apply(lambda x: check_type_room(x, "Shared room"))

data = data.drop(columns=["type"])

host_response = ['nan', 'within an hour', 'within a few hours', 'a few days or more', 'within a day']


def clean_host_response_time(entry):
    if is_nan(entry):
        return 0
    return host_response.index(entry)


data["host_response_time"] = data["host_response_time"].apply(clean_host_response_time)


def clean_review():
    for i in range(len(data)):
        if type(data['reviews'][i]) == str and (data['reviews'][i] == 'eviews' or data['reviews'][i] == 'eview'):
            data.loc[i, 'reviews'] = int(data.loc[i, 'score'])
            data.loc[i, 'score'] = math.nan


clean_review()


def check_nan(data_check):
    for i in range(len(data_check)):
        if is_nan(data_check[i]):
            print("AA", i)


check_nan(data["price"])


def clean_price(entry):
    if is_nan(entry):
        return 0
    return entry[1:]


data["price"] = data["price"].apply(clean_price)
data["service_fee"] = data["service_fee"].apply(clean_price)
data["cleaning_fee"] = data["cleaning_fee"].apply(clean_price)


def remove_percentage_character(entry):
    if is_nan(entry):
        return 30
    return entry[:-1]


data["host_response_rate"] = data["host_response_rate"].apply(remove_percentage_character)


def fill_by_zero(entry):
    if is_nan(entry):
        return 0
    return entry


data["score"] = data["score"].apply(fill_by_zero)
data["reviews"] = data["reviews"].apply(fill_by_zero)
data["bedroom"] = data["bedroom"].apply(fill_by_zero)
data["bed"] = data["bed"].apply(fill_by_zero)
data["bath"] = data["bath"].apply(fill_by_zero)
data["score_cleanliness"] = data["score_cleanliness"].apply(fill_by_zero)
data["score_accuracy"] = data["score_accuracy"].apply(fill_by_zero)
data["score_communication"] = data["score_communication"].apply(fill_by_zero)
data["score_location"] = data["score_location"].apply(fill_by_zero)
data["score_check_in"] = data["score_check_in"].apply(fill_by_zero)
data["score_value"] = data["score_value"].apply(fill_by_zero)
data["host_reviews"] = data["host_reviews"].apply(fill_by_zero)
data["shared_bath"] = data["shared_bath"].apply(fill_by_zero)
data["private_bath"] = data["private_bath"].apply(fill_by_zero)


# data["half-bath"] = data["half-bath"].apply(fill_by_zero)

def format_large_guest(entry):
    if entry == '16+':
        return 16
    return entry


data["guest"] = data["guest"].apply(format_large_guest)


def format_amentity(entry):
    if is_nan(entry):
        return 0
    return 1


data["amentity_wifi"] = data["amentity_wifi"].apply(format_amentity)
data["amentity_dryer"] = data["amentity_dryer"].apply(format_amentity)
data["amentity_essentials"] = data["amentity_essentials"].apply(format_amentity)
data["amentity_hangers"] = data["amentity_hangers"].apply(format_amentity)
data["amentity_hair_dryer"] = data["amentity_hair_dryer"].apply(format_amentity)
data["amentity_washer"] = data["amentity_washer"].apply(format_amentity)
data["amentity_iron"] = data["amentity_iron"].apply(format_amentity)
data["amentity_air_conditioning"] = data["amentity_air_conditioning"].apply(format_amentity)
data["amentity_elevator"] = data["amentity_elevator"].apply(format_amentity)
data["amentity_kitchen"] = data["amentity_kitchen"].apply(format_amentity)
data["amentity_cable_tv"] = data["amentity_cable_tv"].apply(format_amentity)
data["amentity_tv"] = data["amentity_tv"].apply(format_amentity)
data["amentity_laptop-friendly_workspace"] = data["amentity_laptop-friendly_workspace"].apply(format_amentity)
data["amentity_first_aid_kit"] = data["amentity_first_aid_kit"].apply(format_amentity)
data["amentity_indoor_fireplace"] = data["amentity_indoor_fireplace"].apply(format_amentity)
data["amentity_heating"] = data["amentity_heating"].apply(format_amentity)
data["amentity_carbon_monoxide_alarm"] = data["amentity_carbon_monoxide_alarm"].apply(format_amentity)
data["amentity_free_parking_on_premises"] = data["amentity_free_parking_on_premises"].apply(format_amentity)
data["amentity_breakfast"] = data["amentity_breakfast"].apply(format_amentity)
data["amentity_hot_tub"] = data["amentity_hot_tub"].apply(format_amentity)
data["amentity_fire_extinguisher"] = data["amentity_fire_extinguisher"].apply(format_amentity)
data["amentity_gym"] = data["amentity_gym"].apply(format_amentity)
data["amentity_pool"] = data["amentity_pool"].apply(format_amentity)
data["amentity_smoke_alarm"] = data["amentity_smoke_alarm"].apply(format_amentity)
data["amentity_free_street_parking"] = data["amentity_free_street_parking"].apply(format_amentity)
data["amentity_private_entrance"] = data["amentity_private_entrance"].apply(format_amentity)
data["amentity_room-darkening_shades"] = data["amentity_room-darkening_shades"].apply(format_amentity)
data["amentity_building_staff"] = data["amentity_building_staff"].apply(format_amentity)
data["amentity_paid_parking_off_premises"] = data["amentity_paid_parking_off_premises"].apply(format_amentity)
data["amentity_high_chair"] = data["amentity_high_chair"].apply(format_amentity)
data["amentity_bathtub"] = data["amentity_bathtub"].apply(format_amentity)
data["amentity_private_living_room"] = data["amentity_private_living_room"].apply(format_amentity)
data["amentity_lock_on_bedroom_door"] = data["amentity_lock_on_bedroom_door"].apply(format_amentity)


def convert_boolean_to_int(entry):
    if entry:
        return 1
    return 0


data["is_host_verified"] = data["is_host_verified"].apply(convert_boolean_to_int)
data["is_superhost"] = data["is_superhost"].apply(convert_boolean_to_int)
data["no_pets"] = data["no_pets"].apply(convert_boolean_to_int)
data["can_parties"] = data["can_parties"].apply(convert_boolean_to_int)
data["can_smoking"] = data["can_smoking"].apply(convert_boolean_to_int)

data.to_csv("../data/data_cleaned.csv")
