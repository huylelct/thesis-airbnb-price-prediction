import { Form, InputNumber, Button, Row, Col, Select, Card } from 'antd';
import React, { useState } from 'react';
import './App.scss';
import axios from "axios";

const layout = {
  labelCol: {
    span: 12,
  },
  wrapperCol: {
    span: 12,
  },
};
const { Option } = Select;
const tailLayout = {
  wrapperCol: { offset: 5, span: 14 },
};
var children: any = [];
children.push(<Option value="amentity_wifi">Wifi</Option>);
children.push(<Option value="amentity_dryer">Dryer</Option>);
children.push(<Option value="amentity_essentials">Essentials</Option>);
children.push(<Option value="amentity_hangers">Hangers</Option>);
children.push(<Option value="amentity_hair_dryer">Har dryer</Option>);
children.push(<Option value="amentity_washer">Washer</Option>);
children.push(<Option value={"amentity_iron"}>Iron</Option>);
children.push(<Option value={"amentity_air_conditioning"}>Air conditioning</Option>);
children.push(<Option value={"is_host_verified"}>Host verified</Option>);
children.push(<Option value={"is_superhost"}>Superhost</Option>);
children.push(<Option value={"no_pets"}>No pets</Option>);
children.push(<Option value={"can_parties"}>Can parties</Option>);
children.push(<Option value={"can_smoking"}>Can smoking</Option>);
children.push(<Option value={"amentity_elevator"}>Elevator</Option>);
children.push(<Option value={"amentity_kitchen"}>Kitchen</Option>);
children.push(<Option value={"amentity_cable_tv"}>Cable tv</Option>);
children.push(<Option value={"amentity_tv"}>Tv</Option>);
children.push(<Option value={"amentity_laptop_friendly_workspace"}>Laptop workspace</Option>);
children.push(<Option value={"amentity_first_aid_kit"}>First air kit</Option>);
children.push(<Option value={"amentity_indoor_fireplace"}>Indoor fireplace</Option>);
children.push(<Option value={"amentity_heating"}>Heating</Option>);
children.push(<Option value={"amentity_carbon_monoxide_alarm"}>Carbon monoxide alarm</Option>);
children.push(<Option value={"amentity_free_parking_on_premises"}>Free parking on premises</Option>);
children.push(<Option value={"amentity_paid_parking_off_premises"}>Paid parking off premises</Option>);

children.push(<Option value={"amentity_breakfast"}>Breakfast</Option>);
children.push(<Option value={"amentity_fire_extinguisher"}>Fire extinguisher</Option>);
children.push(<Option value={"amentity_hot_tub"}>Hot tub</Option>);
children.push(<Option value={"amentity_gym"}>Gym</Option>);
children.push(<Option value={"amentity_pool"}>Pool</Option>);
children.push(<Option value={"amentity_smoke_alarm"}>Smoke alarm</Option>);
children.push(<Option value={"amentity_free_street_parking"}>Free street parking</Option>);
children.push(<Option value={"amentity_private_entrance"}>Private entrance</Option>);
children.push(<Option value={"amentity_room_darkening_shades"}>Room darkening shades</Option>);
children.push(<Option value={"amentity_building_staff"}>Building staff</Option>);
children.push(<Option value={"amentity_high_chair"}>High chair</Option>);
children.push(<Option value={"amentity_bathtub"}>Bathtub</Option>);
children.push(<Option value={"amentity_private_living_room"}>Private living room</Option>);
children.push(<Option value={"amentity_lock_on_bedroom_door"}>Lock on bedroom door</Option>);
children.push(<Option value={"is_private_room"}>Private room</Option>);
children.push(<Option value={"is_hotel_room"}>Hotel room</Option>);
children.push(<Option value={"is_entire_home"}>Entire home</Option>);

children.push(<Option value={"is_shared_room"}>Shared room</Option>);
var children1: any = [];
children1.push(<Option value="is_dong_nai">Dong Nai</Option>);
children1.push(<Option value="is_binh_duong">Binh Duong</Option>);
children1.push(<Option value="is_ngoai_tinh">Out city</Option>);
children1.push(<Option value="is_phu_nhuan">Phu Nhuan</Option>);
children1.push(<Option value="is_tan_phu">Tan Phu</Option>);
children1.push(<Option value="is_binh_tan">Binh Tan</Option>);
children1.push(<Option value="is_quan_4">District 4</Option>);
children1.push(<Option value="is_can_gio">Can Gio</Option>);
children1.push(<Option value="is_nha_be">Nha Be</Option>);
children1.push(<Option value="is_quan_5">District 5 </Option>);
children1.push(<Option value="is_go_vap">Go Vap</Option>);
children1.push(<Option value="is_binh_chanh">Binh Thanh</Option>);
children1.push(<Option value="is_quan_12">District 12</Option>);
children1.push(<Option value="is_tan_binh">Tan Binh</Option>);
children1.push(<Option value="is_quan_3">District 3</Option>);
children1.push(<Option value="is_quan_7">District 7</Option>);
children1.push(<Option value="is_hoc_mon">Hoc Mon</Option>);
children1.push(<Option value="is_quan_9">District 9</Option>);
children1.push(<Option value="is_quan_6">District 6</Option>);
children1.push(<Option value="is_quan_2">District 2</Option>);
children1.push(<Option value="is_quan_8">District 8</Option>);
children1.push(<Option value="is_quan_10">District 10</Option>);
children1.push(<Option value="is_quan_1">District 1</Option>);
children1.push(<Option value="is_quan_11">District 11</Option>);
children1.push(<Option value="is_cu_chi">Cu Chi</Option>);
children1.push(<Option value="is_thu_duc">Thu Duc</Option>);
children1.push(<Option value="is_binh_thanh">Binh Thanh</Option>);



const Demo = () => {
  const [price, setPrice] = useState(0);

  const onFinish = (values: any) => {

    // values.amentity_wifi = 0;
    // values.amentity_dryer = 0;
    // values.amentity_essentials = 0;
    // values.amentity_hangers = 0;
    // values.amentity_hair_dryer = 0;
    // values.amentity_washer = 0;
    // values.amentity_iron = 0;
    // values.amentity_air_conditioning = 0;
    // values.is_host_verified = 0;
    // values.is_superhost = 0;
    // values.no_pets = 0;
    // values.can_parties = 0;
    // values.can_smoking = 0;
    // values.amentity_elevator = 0;
    // values.amentity_kitchen = 0;
    // values.amentity_cable_tv = 0;
    // values.amentity_tv = 0;
    // values.amentity_laptop_friendly_workspace = 0;
    // values.amentity_first_aid_kit = 0;
    // values.amentity_indoor_fireplace = 0;
    // values.amentity_heating = 0;
    // values.amentity_carbon_monoxide_alarm = 0;
    // values.amentity_free_parking_on_premises = 0;
    // values.amentity_paid_parking_off_premises = 0;

    // values.amentity_breakfast = 0;
    // values.amentity_fire_extinguisher = 0;
    // values.amentity_hot_tub = 0;
    // values.amentity_gym = 0;
    // values.amentity_pool = 0;
    // values.amentity_smoke_alarm = 0;
    // values.amentity_free_street_parking = 0;
    // values.amentity_private_entrance = 0;
    // values.amentity_room_darkening_shades = 0;
    // values.amentity_building_staff = 0;
    // values.amentity_high_chair = 0;
    // values.amentity_bathtub = 0;
    // values.amentity_private_living_room = 0;
    // values.amentity_lock_on_bedroom_door = 0;
    // values.is_entire_home = 0;
    // values.is_shared_room = 0;
    // values.is_private_room = 0;
    // values.is_hotel_room = 0;
    // values.is_dong_nai = 0;
    // values.is_binh_duong = 0;
    // values.is_ngoai_tinh = 0;
    // values.is_phu_nhuan = 0;
    // values.is_tan_phu = 0;
    // values.is_binh_tan = 0;
    // values.is_quan_4 = 0;
    // values.is_can_gio = 0;
    // values.is_nha_be = 0;
    // values.is_quan_5 = 0;
    // values.is_go_vap = 0;
    // values.is_binh_chanh = 0;
    // values.is_quan_12 = 0;
    // values.is_tan_binh = 0;
    // values.is_quan_3 = 0;
    // values.is_quan_7 = 0;
    // values.is_hoc_mon = 0;
    // values.is_quan_9 = 0;
    // values.is_quan_6 = 0;
    // values.is_quan_2 = 0;
    // values.is_quan_8 = 0;
    // values.is_quan_10 = 0;
    // values.is_quan_1 = 0;
    // values.is_quan_11 = 0;
    // values.is_cu_chi = 0;
    // values.is_thu_duc = 0;
    // values.is_binh_thanh = 0;

    if (values.location != undefined) {
      values[values.location] = 1;
      console.log(values[values.location])
      values.location = undefined;
    }
    console.log('Success:', values);
    if (values.option != undefined) {
      values.option.forEach((x: any) => values[x] = 1);
      values.option = undefined;

    }
    for (var name in values) {
      if (values[name] == null) values[name] = undefined
    }
    const data = {
      data: values
    }
    axios.post('https://backend-price-predict.herokuapp.com/employees', data)
      .then(function (response) {
        console.log(response.data);
        setPrice(response.data)
      })
  };
  const [showResult, setshowResult] = useState(false);

  const onFinishFailed = (errorInfo: any) => {
    console.log('Failed:', errorInfo);
  };

  return (
    <div className="main">

      <h1 className="title">{process.env.REACT_APP_API_URL}</h1>
      <Form {...layout}
        name="basic"
        initialValues={{ remember: true }}
        onFinish={onFinish}
        onFinishFailed={onFinishFailed}
      >
        <Row>
          <Col span={6}>
            <Form.Item
              label="Score"
              name="score"
            //rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Reviews"
              name="reviews"
            //rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Number guests"
              name="guest"
            //rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Number bedrooms"
              name="bedroom"
            //rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
        </Row><Row>
          <Col span={6}>
            <Form.Item
              label="Number beds"
              name="bed"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Number baths"
              name="bath"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>

          <Col span={6}>
            <Form.Item
              label="Score cleanliness"
              name="score_cleanliness"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Score accuracy"
              name="score_accuracy"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Score communication"
              name="score_communication"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Score location"
              name="score_location"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Score value"
              name="score_value"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>



          <Col span={6}>
            <Form.Item
              label="Score check in"
              name="score_check_in"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
        </Row><Row>


          <Col span={6}>
            <Form.Item
              label="Service fee"
              name="service_fee"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Cleaning fee"
              name="cleaning_fee"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Host reviews"
              name="host_reviews"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Host reponse rate"
              name="host_response_rate"

            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>




          <Col span={6}>
            <Form.Item
              label="Host response time"
              name="host_response_time"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          {/* <Col span={6}>
            <Form.Item  
              label="Number share bath"
              name="shared_bath"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>

          <Col span={6}>
            <Form.Item  
              label="Number private bath"
              name="private_bath"
            ////rules={[{ required: true, message: 'Please input this field' }]}
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>*/}
        </Row><Row>
          <Col span={12}>
            <Form.Item
              label="Option"
              name="option"
            >
              <Select
                mode="multiple"
                style={{ width: '100%' }}
              >
                {children}
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="Location"
              name="location"
            >
              <Select
                style={{ width: '100%' }}
              >
                {children1}
              </Select>
            </Form.Item>
          </Col>
        </Row>
        <Form.Item  {...tailLayout}>
          <Button className="button" type="default" htmlType="submit">
            Submit
        </Button>
        </Form.Item>
      </Form>
      {price !== 0 && <Card >
        Price: <span className="red">{price} $</span>
      </Card>}
    </div >
  );
};

export default Demo;