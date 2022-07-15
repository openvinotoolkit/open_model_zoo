/*
C++ MJPEG over HTTP Library
https://github.com/nadjieb/cpp-mjpeg-streamer

MIT License

Copyright (c) 2020-2021 Muhammad Kamal Nadjieb

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <csignal>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sstream>
#include <string>
#include <unordered_map>

namespace ncam {

struct HTTPMessage {
    HTTPMessage() = default;
    HTTPMessage(const std::string& message) { parse(message); }

    std::string start_line;
    std::unordered_map<std::string, std::string> headers;
    std::string body;

    std::string serialize() const {
        const std::string delimiter = "\r\n";
        std::stringstream stream;

        stream << start_line << delimiter;

        for (const auto& header : headers) {
            stream << header.first << ": " << header.second << delimiter;
        }

        stream << delimiter << body;

        return stream.str();
    }

    void parse(const std::string& message) {
        const std::string delimiter = "\r\n";
        const std::string body_delimiter = "\r\n\r\n";

        start_line = message.substr(0, message.find(delimiter));

        auto raw_headers = message.substr(
            message.find(delimiter) + delimiter.size(),
            message.find(body_delimiter) - message.find(delimiter));

        while (raw_headers.find(delimiter) != std::string::npos) {
            auto header = raw_headers.substr(0, raw_headers.find(delimiter));
            auto key = header.substr(0, raw_headers.find(':'));
            auto value = header.substr(raw_headers.find(':') + 1, raw_headers.find(delimiter));
            while (value[0] == ' ') {
                value = value.substr(1);
            }
            headers[std::string(key)] = std::string(value);
            raw_headers = raw_headers.substr(raw_headers.find(delimiter) + delimiter.size());
        }

        body = message.substr(message.find(body_delimiter) + body_delimiter.size());
    }

    std::string target() const {
        std::string result(start_line.c_str() + start_line.find(' ') + 1);
        result = result.substr(0, result.find(' '));
        return std::string(result);
    }

    std::string method() const { return start_line.substr(0, start_line.find(' ')); }
};

}  // namespace ncam


namespace ncam {

constexpr int NUM_SEND_MUTICES = 100;

class MJPEGStreamer {
public:
    MJPEGStreamer() = default;
    virtual ~MJPEGStreamer() { stop(); }

    MJPEGStreamer(MJPEGStreamer&&) = delete;
    MJPEGStreamer(const MJPEGStreamer&) = delete;
    MJPEGStreamer& operator=(MJPEGStreamer&&) = delete;
    MJPEGStreamer& operator=(const MJPEGStreamer&) = delete;

    void start(int port, int num_workers = 1) {
        ::signal(SIGPIPE, SIG_IGN);
        master_socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
        panicIfUnexpected(master_socket_ < 0, "ERROR: socket not created\n");

        int yes = 1;
        auto res = ::setsockopt(
            master_socket_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&yes), sizeof(yes));
        panicIfUnexpected(res < 0, "ERROR: setsocketopt SO_REUSEADDR\n");

        address_.sin_family = AF_INET;
        address_.sin_addr.s_addr = INADDR_ANY;
        address_.sin_port = htons(port);
        res = ::bind(
            master_socket_, reinterpret_cast<struct sockaddr*>(&address_), sizeof(address_));
        panicIfUnexpected(res < 0, "ERROR: bind\n");

        res = ::listen(master_socket_, 5);
        panicIfUnexpected(res < 0, "ERROR: listen\n");

        for (auto i = 0; i < num_workers; ++i) {
            workers_.emplace_back(worker());
        }

        thread_listener_ = std::thread(listener());
    }

    void stop() {
        if (isAlive()) {
            std::unique_lock<std::mutex> lock(payloads_mutex_);
            master_socket_ = -1;
            condition_.notify_all();
        }

        if (!workers_.empty()) {
            for (auto& w : workers_) {
                if (w.joinable()) {
                    w.join();
                }
            }
            workers_.clear();
        }

        if (!path2clients_.empty()) {
            for (auto& p2c : path2clients_) {
                for (auto sd : p2c.second) {
                    ::close(sd);
                }
            }
            path2clients_.clear();
        }

        if (thread_listener_.joinable()) {
            thread_listener_.join();
        }
    }

    void publish(const std::string& path, const std::string& buffer) {
        std::vector<int> clients;
        {
            std::unique_lock<std::mutex> lock(clients_mutex_);
            if ((path2clients_.find(path) == path2clients_.end())
                || (path2clients_[path].empty())) {
                return;
            }
            clients = path2clients_[path];
        }

        for (auto i : clients) {
            std::unique_lock<std::mutex> lock(payloads_mutex_);
            payloads_.emplace(Payload{buffer, path, i});
            condition_.notify_one();
        }
    }

    bool isAlive() {
        std::unique_lock<std::mutex> lock(payloads_mutex_);
        return master_socket_ > 0;
    }

private:
    struct Payload {
        std::string buffer;
        std::string path;
        int sd;
    };

    int master_socket_ = -1;
    struct sockaddr_in address_;

    std::thread thread_listener_;
    std::mutex clients_mutex_;
    std::mutex payloads_mutex_;
    std::array<std::mutex, NUM_SEND_MUTICES> send_mutices_;
    std::condition_variable condition_;

    std::vector<std::thread> workers_;
    std::queue<Payload> payloads_;
    std::unordered_map<std::string, std::vector<int>> path2clients_;

    std::function<void()> worker() {
        return [this]() {
            while (this->isAlive()) {
                Payload payload;

                {
                    std::unique_lock<std::mutex> lock(this->payloads_mutex_);
                    this->condition_.wait(lock, [this]() {
                        return this->master_socket_ < 0 || !this->payloads_.empty();
                    });
                    if ((this->master_socket_ < 0) && (this->payloads_.empty())) {
                        return;
                    }
                    payload = std::move(this->payloads_.front());
                    this->payloads_.pop();
                }

                HTTPMessage res;
                res.start_line = "--boundarydonotcross";
                res.headers["Content-Type"] = "image/jpeg";
                res.headers["Content-Length"] = std::to_string(payload.buffer.size());
                res.body = payload.buffer;

                auto res_str = res.serialize();

                int n;
                {
                    std::unique_lock<std::mutex> lock(
                        this->send_mutices_.at(payload.sd % NUM_SEND_MUTICES));
                    n = ::write(payload.sd, res_str.c_str(), res_str.size());
                }

                if (n < static_cast<int>(res_str.size())) {
                    std::unique_lock<std::mutex> lock(this->clients_mutex_);
                    auto& p2c = this->path2clients_[payload.path];
                    if (std::find(p2c.begin(), p2c.end(), payload.sd) != p2c.end()) {
                        p2c.erase(std::remove(p2c.begin(), p2c.end(), payload.sd), p2c.end());
                        ::close(payload.sd);
                    }
                }
            }
        };
    }

    std::function<void()> listener() {
        return [this]() {
            HTTPMessage bad_request_res;
            bad_request_res.start_line = "HTTP/1.1 400 Bad Request";

            HTTPMessage shutdown_res;
            shutdown_res.start_line = "HTTP/1.1 200 OK";

            HTTPMessage method_not_allowed_res;
            method_not_allowed_res.start_line = "HTTP/1.1 405 Method Not Allowed";

            HTTPMessage init_res;
            init_res.start_line = "HTTP/1.1 200 OK";
            init_res.headers["Connection"] = "close";
            init_res.headers["Cache-Control"]
                = "no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0";
            init_res.headers["Pragma"] = "no-cache";
            init_res.headers["Content-Type"]
                = "multipart/x-mixed-replace; boundary=boundarydonotcross";

            auto bad_request_res_str = bad_request_res.serialize();
            auto shutdown_res_str = shutdown_res.serialize();
            auto method_not_allowed_res_str = method_not_allowed_res.serialize();
            auto init_res_str = init_res.serialize();

            int addrlen = sizeof(this->address_);

            fd_set fd;
            FD_ZERO(&fd);

            auto master_socket = this->master_socket_;

            while (this->isAlive()) {
                struct timeval to;
                to.tv_sec = 1;
                to.tv_usec = 0;

                FD_SET(this->master_socket_, &fd);

                if (select(this->master_socket_ + 1, &fd, nullptr, nullptr, &to) > 0) {
                    auto new_socket = ::accept(
                        this->master_socket_, reinterpret_cast<struct sockaddr*>(&(this->address_)),
                        reinterpret_cast<socklen_t*>(&addrlen)
                    );
                    if (this->isAlive()) {
                        this->panicIfUnexpected(new_socket < 0, "ERROR: accept\n");
                    } else {
                        return;
                    }

                    std::string buff(4096, 0);
                    this->readBuff(new_socket, &buff[0], buff.size());

                    HTTPMessage req(buff);

                    if (req.method() != "GET") {
                        this->writeBuff(
                            new_socket, method_not_allowed_res_str.c_str(),
                            method_not_allowed_res_str.size());
                        ::close(new_socket);
                        continue;
                    }

                    this->writeBuff(new_socket, init_res_str.c_str(), init_res_str.size());

                    std::unique_lock<std::mutex> lock(this->clients_mutex_);
                    this->path2clients_[req.target()].push_back(new_socket);
                }
            }

            ::shutdown(master_socket, 2);
        };
    }

    static void panicIfUnexpected(bool condition, const std::string& message) {
        if (condition) {
            throw std::runtime_error(message);
        }
    }

    static void readBuff(int fd, void* buf, size_t nbyte) {
        panicIfUnexpected(::read(fd, buf, nbyte) < 0, "ERROR: read\n");
    }

    static void writeBuff(int fd, const void* buf, size_t nbyte) {
        panicIfUnexpected(::write(fd, buf, nbyte) < 0, "ERROR: write\n");
    }
};
}  // namespace ncam
