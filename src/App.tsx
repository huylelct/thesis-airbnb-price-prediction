import { Form, InputNumber, Button, Row, Col, Select, Card, Tag } from 'antd';
import React, { useState } from 'react';
import './App.scss';
import axios from "axios";

const layout = {
  labelCol: {
    span: 14,
  },
  wrapperCol: {
    span: 10,
  },
};

const { Option } = Select;
const tailLayout = {
  wrapperCol: { offset: 5, span: 14 },
};

var children: any = [];
var children1: any = [];

var children2: any = [];
var children3: any = [];
var children4: any = [];

children.push(<Option value="amentity_wifi">Wifi</Option>);
children.push(<Option value="amentity_dryer">Dryer</Option>);
children.push(<Option value="amentity_essentials">Essentials</Option>);
children.push(<Option value="amentity_hangers">Hangers</Option>);
children.push(<Option value="amentity_hair_dryer">Har dryer</Option>);
children.push(<Option value="amentity_washer">Washer</Option>);
children.push(<Option value={"amentity_iron"}>Iron</Option>);
children.push(<Option value={"amentity_air_conditioning"}>Air conditioning</Option>);
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

children2.push(<Option value={"is_private_room"}>Private room</Option>);
children2.push(<Option value={"is_hotel_room"}>Hotel room</Option>);
children2.push(<Option value={"is_entire_home"}>Entire home</Option>);
children2.push(<Option value={"is_shared_room"}>Shared room</Option>);

children3.push(<Option value={"is_host_verified"}>Host verified</Option>);
children3.push(<Option value={"is_superhost"}>Superhost</Option>);

children4.push(<Option value={1}>Within an hour</Option>);
children4.push(<Option value={2}>Within a few hours</Option>);
children4.push(<Option value={3}>Within a day</Option>);
children4.push(<Option value={4}>A few days or more</Option>);

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

    if (values.location != undefined) {
      values[values.location] = 1;
      values.location = undefined;
    }

    if (values.option != undefined) {
      values.option.forEach((x: any) => values[x] = 1);
      values.option = undefined;
    }

    if (values.roomtype != undefined) {
      values[values.roomtype] = 1;
      values.roomtype = undefined;
    }

    if (values.hostinfor != undefined) {
      values.hostinfor.forEach((x: any) => values[x] = 1);
      values.hostinfor = undefined;
    }

    if (values.option != undefined) {
      values.option.forEach((x: any) => values[x] = 1);
      values.option = undefined;
    }

    for (var name in values) {
      if (values[name] == null) values[name] = undefined
    }

    for (var name in values) {
      if (values[name] == undefined && name == 'score') values[name] = 4.7;
      if (values[name] == undefined && name == 'score_cleanliness') values[name] = values.score;
      if (values[name] == undefined && name == 'score_accuracy') values[name] = values.score;
      if (values[name] == undefined && name == 'score_location') values[name] = values.score;
      if (values[name] == undefined && name == 'score_check_in') values[name] = values.score;
      if (values[name] == undefined && name == 'score_communication') values[name] = values.score;
      if (values[name] == undefined && name == 'score_value') values[name] = values.score;
      if (values[name] == undefined && name == 'reviews') values[name] = 25;
      if (values[name] == undefined && name == 'host_reviews') values[name] = 100;
      if (values[name] == undefined && name == 'host_response_rate') values[name] = 70;
      if (values[name] == undefined && name == 'host_response_time') values[name] = 1;
      if (values[name] == undefined && name == 'guest') values[name] = 3.5;
      if (values[name] == undefined && name == 'bedroom') values[name] = 1.5;
      if (values[name] == undefined && name == 'bed') values[name] = 2.5;
      if (values[name] == undefined && name == 'bath') values[name] = 1.5;
      if (values[name] == undefined && name == 'service_fee') values[name] = 6;
      if (values[name] == undefined && name == 'cleaning_fee') values[name] = 4;
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
          <Tag className="tag" color="geekblue">Basic information</Tag>
        </Row>
        <Row>
          <Col span={6}>
            <Form.Item
              label="Reviews"
              name="reviews"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Number guests"
              name="guest"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Number bedrooms"
              name="bedroom"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Number beds"
              name="bed"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Number baths"
              name="bath"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Service fee"
              name="service_fee"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Cleaning fee"
              name="cleaning_fee"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Room type"
              name="roomtype"
            >
              <Select
                style={{ width: '100%' }}
              >
                {children2}
              </Select>
            </Form.Item>
          </Col>
        </Row>
        <Row>
          <Tag className="tag" color="geekblue">Score</Tag>
        </Row>
        <Row>
          <Col span={6}>
            <Form.Item
              label="Score"
              name="score"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Score cleanliness"
              name="score_cleanliness"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Score accuracy"
              name="score_accuracy"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Score communication"
              name="score_communication"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Score location"
              name="score_location"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Score value"
              name="score_value"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Score check in"
              name="score_check_in"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
        </Row>
        <Row>
          <Tag className="tag" color="geekblue">Host information</Tag>
        </Row>
        <Row>

          <Col span={6}>
            <Form.Item
              label="Host reviews"
              name="host_reviews"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Host reponse rate"
              name="host_response_rate"
            >
              <InputNumber defaultValue={0} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item
              label="Host response time"
              name="host_response_time"
            >
              <Select
                style={{ width: '100%' }}
              >
                {children4}
              </Select>
            </Form.Item>
          </Col>
          <Col span={6}>

            <Form.Item
              label="Host information"
              name="hostinfor"
            >
              <Select
                mode="multiple"
                style={{ width: '100%' }}
              >
                {children3}
              </Select>
            </Form.Item>
          </Col>
        </Row>
        <Row>
          <Tag className="tag" color="geekblue">Other information</Tag>
        </Row>
        <Row>
          <Col span={12}>
            <Form.Item
              label="Amenity"
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
      {
        price !== 0 && <Card >
          Price: <span className="red">{price} $</span>
        </Card>
      }
    </div >
  );
};

export default Demo;