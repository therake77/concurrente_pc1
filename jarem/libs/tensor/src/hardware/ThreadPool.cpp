#include <hardware/ThreadPool.hpp>

using namespace MyTensors::Hardware;

ThreadPool::ThreadPool(std::size_t _n)
    : n_workers(_n),
    tasks(_n),
    mutexes(_n),
    signal_valid_task(_n),
    is_valid_task(_n),
    remaining(0)
{
    this->workers.reserve(_n);
    for(std::size_t i = 0; i < _n; i++){
        this->workers.emplace_back(&ThreadPool::worker_loop, this, i);
    }
}

void ThreadPool::worker_loop(std::size_t idx){
    for(;;){
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(this->mutexes[idx]);
            this->signal_valid_task[idx].wait(lock, [this,idx]{
                return this->is_valid_task[idx].load() || this->stop.load();
            });
            if(this->stop.load() && !this->is_valid_task[idx].load()){ return; }
            this->is_valid_task[idx] = false;
            //Only read the task once we've confirmed (under the lock) that it's actually valid
            task = this->tasks[idx];
        }
        task();
        this->remaining--;
        {
            std::lock_guard<std::mutex> done_lock(this->done_mutex);
        }
        this->done_signal.notify_one();
    }
}

void ThreadPool::synchronize(){
    std::unique_lock<std::mutex> lock(this->done_mutex);
    this->done_signal.wait(lock, [this](){ return this->remaining <= 0; });
    return;
}

void ThreadPool::reset(){
    this->remaining = 0;
    for(std::size_t i =0 ; i < this->n_workers; i++){
        {
            std::lock_guard<std::mutex> lock(mutexes[i]);
            this->is_valid_task[i] = false;
        }
    }

}

ThreadPool::~ThreadPool(){
    this->stop = true;
    for(std::size_t i = 0; i < this->n_workers; i++){
        this->signal_valid_task[i].notify_one();
    }
    for(auto& worker : this->workers){
        worker.join();
    }
}
