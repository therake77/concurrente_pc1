#pragma once
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <cstddef>
#include <atomic>

namespace MyTensors::Hardware{

    class ThreadPool{
    private:
        std::size_t n_workers;
        std::vector<std::thread> workers;
        std::vector<std::function<void()>> tasks;
        std::atomic<int> remaining;

        std::vector<std::condition_variable> signal_valid_task;
        std::vector<std::atomic<bool>> is_valid_task;
        std::vector<std::mutex> mutexes;

        std::condition_variable done_signal;
        std::mutex done_mutex;

        std::atomic<bool> stop{false};

        void worker_loop(std::size_t idx);

    public:
        explicit ThreadPool(std::size_t _n);
        ~ThreadPool();

        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;

        std::size_t size() const { return workers.size(); }

        template<typename F, typename... Args>
        void assign_to(std::size_t thread_idx, F&& f, Args&&... args){
            if(this->workers.size()<= thread_idx){ return; }

            using ReturnType = std::invoke_result_t<F, Args...>;

            auto task = std::make_shared<std::packaged_task<ReturnType()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );

            {
                std::lock_guard<std::mutex> lock(this->mutexes[thread_idx]);
                this->is_valid_task[thread_idx] = true;
                this->tasks[thread_idx] = ([task](){ (*task)(); });
                this->remaining++;
            }
            this->signal_valid_task[thread_idx].notify_one();
            return;
        }

        void reset();
        void synchronize();

    };

};
