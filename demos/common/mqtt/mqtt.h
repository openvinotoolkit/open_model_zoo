#ifndef MQTT_H
#define MQTT_H

#include <mosquittopp.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <string>

#define MAX_PAYLOAD 50
#define DEFAULT_KEEP_ALIVE 60
#define GENERAL_TOPIC "OpenVINO"
class mqtt : public mosqpp::mosquittopp {
private:
    const char * broker;
    const char * id;
    std::string topic;
    int port;
    int keepalive;
    void on_connect(int rc);
    void on_disconnect(int rc);
    void on_publish(int mid);

public:
    mqtt(const char *id, const char * application_topic,const char * broker, int port);
    ~mqtt();
    bool send_message(const char * message);
};
#endif //MQTT_H
