// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/executor_manager.hpp"

#include "openvino/core/parallel.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO || OV_THREAD == OV_THREAD_TBB_ADAPTIVE
#    if (TBB_INTERFACE_VERSION < 12000)
#        include <tbb/task_scheduler_init.h>
#    else
#        include <oneapi/tbb/global_control.h>
#    endif
#endif

#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace ov {
namespace threading {
namespace {
class ExecutorManagerImpl : public ExecutorManager {
public:
    ~ExecutorManagerImpl();
    std::shared_ptr<ov::threading::ITaskExecutor> get_executor(const std::string& id) override;
    std::shared_ptr<ov::threading::IStreamsExecutor> get_idle_cpu_streams_executor(
        const ov::threading::IStreamsExecutor::Config& config) override;
    size_t get_executors_number() const override;
    size_t get_idle_cpu_streams_executors_number() const override;
    void clear(const std::string& id = {}) override;
    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;
    void execute_task_by_streams_executor(ov::hint::SchedulingCoreType core_type, ov::threading::Task task) override;

private:
    void reset_tbb();

    std::unordered_map<std::string, std::shared_ptr<ov::threading::ITaskExecutor>> executors;
    std::vector<std::pair<ov::threading::IStreamsExecutor::Config, std::shared_ptr<ov::threading::IStreamsExecutor>>>
        cpuStreamsExecutors;
    mutable std::mutex streamExecutorMutex;
    mutable std::mutex taskExecutorMutex;
    bool tbbTerminateFlag = false;
    mutable std::mutex global_mutex;
    bool tbbThreadsCreated = false;
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO || OV_THREAD == OV_THREAD_TBB_ADAPTIVE
#    if (TBB_INTERFACE_VERSION < 12000)
    std::shared_ptr<tbb::task_scheduler_init> tbbTaskScheduler = nullptr;
#    else
    std::shared_ptr<oneapi::tbb::task_scheduler_handle> tbbTaskScheduler = nullptr;
#    endif
#endif
};

}  // namespace

ExecutorManagerImpl::~ExecutorManagerImpl() {
    reset_tbb();
}

void ExecutorManagerImpl::set_property(const ov::AnyMap& properties) {
    std::lock_guard<std::mutex> guard(global_mutex);
    for (const auto& it : properties) {
        if (it.first == ov::force_tbb_terminate.name()) {
            tbbTerminateFlag = it.second.as<bool>();
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO || OV_THREAD == OV_THREAD_TBB_ADAPTIVE
            if (tbbTerminateFlag) {
                if (!tbbTaskScheduler) {
#    if (TBB_INTERFACE_VERSION < 12000)
                    tbbTaskScheduler = std::make_shared<tbb::task_scheduler_init>();
#    elif (TBB_INTERFACE_VERSION < 12060)
                    tbbTaskScheduler =
                        std::make_shared<oneapi::tbb::task_scheduler_handle>(oneapi::tbb::task_scheduler_handle::get());
#    else
                    tbbTaskScheduler = std::make_shared<oneapi::tbb::task_scheduler_handle>(tbb::attach{});
#    endif
                }
            } else {
                tbbTaskScheduler = nullptr;
            }
#endif
        }
    }
}
ov::Any ExecutorManagerImpl::get_property(const std::string& name) const {
    std::lock_guard<std::mutex> guard(global_mutex);
    if (name == ov::force_tbb_terminate.name()) {
        return tbbTerminateFlag;
    }
    OPENVINO_THROW("Property ", name, " is not supported.");
}

void ExecutorManagerImpl::reset_tbb() {
    std::lock_guard<std::mutex> guard(global_mutex);
    if (tbbTerminateFlag) {
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO || OV_THREAD == OV_THREAD_TBB_ADAPTIVE
        if (tbbTaskScheduler && tbbThreadsCreated) {
#    if (TBB_INTERFACE_VERSION < 12000)
            tbbTaskScheduler->terminate();
#    else
            tbb::finalize(*tbbTaskScheduler, std::nothrow);
#    endif
        }
        tbbThreadsCreated = false;
        tbbTaskScheduler = nullptr;
#endif
        tbbTerminateFlag = false;
    }
}

std::shared_ptr<ov::threading::ITaskExecutor> ExecutorManagerImpl::get_executor(const std::string& id) {
    std::lock_guard<std::mutex> guard(taskExecutorMutex);
    std::cerr << "[trace] ExecutorManagerImpl::get_executor(\"" << id << "\")" << std::endl;
    auto foundEntry = executors.find(id);
    if (foundEntry == executors.end()) {
        std::cerr << "[trace]   未找到已有executor → 创建新 CPUStreamsExecutor(\"" << id << "\")" << std::endl;
        auto newExec = std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{id});
        tbbThreadsCreated = true;
        executors[id] = newExec;
        std::cerr << "[trace]   executors 总数=" << executors.size() << ", tbbThreadsCreated=true" << std::endl;
        return newExec;
    }
    std::cerr << "[trace]   复用已有executor(\"" << id << "\")" << std::endl;
    return foundEntry->second;
}

std::shared_ptr<ov::threading::IStreamsExecutor> ExecutorManagerImpl::get_idle_cpu_streams_executor(
    const ov::threading::IStreamsExecutor::Config& config) {
    std::lock_guard<std::mutex> guard(streamExecutorMutex);
    for (auto& it : cpuStreamsExecutors) {
        const auto& executor = it.second;
        if (executor.use_count() != 1)
            continue;

        auto& executorConfig = it.first;
        if (executorConfig == config)
            return executor;
    }
    auto newExec = std::make_shared<ov::threading::CPUStreamsExecutor>(config);
    tbbThreadsCreated = true;
    cpuStreamsExecutors.emplace_back(std::make_pair(config, newExec));
    return newExec;
}

size_t ExecutorManagerImpl::get_executors_number() const {
    std::lock_guard<std::mutex> guard(taskExecutorMutex);
    return executors.size();
}

size_t ExecutorManagerImpl::get_idle_cpu_streams_executors_number() const {
    std::lock_guard<std::mutex> guard(streamExecutorMutex);
    return cpuStreamsExecutors.size();
}

void ExecutorManagerImpl::clear(const std::string& id) {
    std::lock_guard<std::mutex> stream_guard(streamExecutorMutex);
    std::lock_guard<std::mutex> task_guard(taskExecutorMutex);
    if (id.empty()) {
        executors.clear();
        cpuStreamsExecutors.clear();
    } else {
        executors.erase(id);
        cpuStreamsExecutors.erase(std::remove_if(cpuStreamsExecutors.begin(),
                                                 cpuStreamsExecutors.end(),
                                                 [&](std::pair<ov::threading::IStreamsExecutor::Config,
                                                               std::shared_ptr<ov::threading::IStreamsExecutor>>& it) {
                                                     return it.first.get_name() == id;
                                                 }),
                                  cpuStreamsExecutors.end());
    }
}

void ExecutorManagerImpl::execute_task_by_streams_executor(ov::hint::SchedulingCoreType core_type,
                                                           ov::threading::Task task) {
    ov::threading::IStreamsExecutor::Config streamsConfig("StreamsExecutor", 1, 1, core_type, false, true);
    if (!streamsConfig.get_streams_info_table().empty()) {
        auto taskExecutor = std::make_shared<ov::threading::CPUStreamsExecutor>(streamsConfig);
        std::vector<Task> tasks{std::move(task)};
        taskExecutor->run_and_wait(tasks);
    }
}

namespace {

class ExecutorManagerHolder {
    std::mutex _mutex;
    std::weak_ptr<ExecutorManager> _manager;

public:
    ExecutorManagerHolder(const ExecutorManagerHolder&) = delete;
    ExecutorManagerHolder& operator=(const ExecutorManagerHolder&) = delete;

    ExecutorManagerHolder() {
        std::cerr << "[trace] ExecutorManagerHolder 构造 (static局部变量, 进程生命周期只执行一次)" << std::endl;
    }

    std::shared_ptr<ov::threading::ExecutorManager> get() {
        std::lock_guard<std::mutex> lock(_mutex);
        std::cerr << "[trace] ExecutorManagerHolder::get() → weak_ptr::lock() 尝试提升..." << std::endl;
        auto manager = _manager.lock();
        if (!manager) {
            std::cerr << "[trace]   weak_ptr 为空(首次 或 旧manager已销毁) → 创建新 ExecutorManagerImpl" << std::endl;
            _manager = manager = std::make_shared<ExecutorManagerImpl>();
            std::cerr << "[trace]   新 ExecutorManagerImpl 创建完毕, use_count=" << manager.use_count()
                      << " (weak_ptr不增加计数)" << std::endl;
        } else {
            std::cerr << "[trace]   weak_ptr 提升成功(已有manager存活), use_count=" << manager.use_count() << std::endl;
        }
        return manager;
    }
};

}  // namespace

std::shared_ptr<ExecutorManager> executor_manager() {
    std::cerr << "[trace] executor_manager() 被调用" << std::endl;
    static ExecutorManagerHolder executorManagerHolder;
    std::cerr << "[trace] 调用 executorManagerHolder.get()" << std::endl;
    return executorManagerHolder.get();
}

}  // namespace threading
}  // namespace ov
