/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#if defined(__linux__)
#include <sys/sendfile.h>
#endif
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <unordered_set>

#include "tensorflow/core/platform/default/transactional_posix_file_system.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/error.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

// 128KB of copy buffer
constexpr size_t kPosixCopyFileBufferSize = 128 * 1024;

class PosixTransactionToken final {
 public:
  void Add(const string& path) {
    mutex_lock l(mu_);
    files.insert(path);
  }
  void Remove(const string& path) {
    mutex_lock l(mu_);
    files.erase(path);
  }
  void Rename(const string& src, const string& dst) {
    mutex_lock l(mu_);
    files.erase(src);
    files.insert(dst);
  }
  bool Contains(const string& path) {
    mutex_lock l(mu_);
    return files.count(path) > 0;
  }
  mutex mu_;
  std::unordered_set<string> files;
};

// pread() based random-access
class TransactionalPosixRandomAccessFile : public RandomAccessFile {
 private:
  string filename_;
  int fd_;

 public:
  TransactionalPosixRandomAccessFile(const string& fname, int fd)
      : filename_(fname), fd_(fd) {}
  ~TransactionalPosixRandomAccessFile() override {
    if (close(fd_) < 0) {
      LOG(ERROR) << "close() failed: " << strerror(errno);
    }
  }

  Status Name(StringPiece* result) const override {
    *result = filename_;
    return Status::OK();
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    Status s;
    char* dst = scratch;
    while (n > 0 && s.ok()) {
      // Some platforms, notably macs, throw EINVAL if pread is asked to read
      // more than fits in a 32-bit integer.
      size_t requested_read_length;
      if (n > INT32_MAX) {
        requested_read_length = INT32_MAX;
      } else {
        requested_read_length = n;
      }
      ssize_t r =
          pread(fd_, dst, requested_read_length, static_cast<off_t>(offset));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
      } else if (r == 0) {
        s = Status(error::OUT_OF_RANGE, "Read less bytes than requested");
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        s = IOError(filename_, errno);
      }
    }
    *result = StringPiece(scratch, dst - scratch);
    return s;
  }
};

class TransactionalPosixWritableFile : public WritableFile {
 private:
  string filename_;
  FILE* file_;

 public:
  TransactionalPosixWritableFile(const string& fname, FILE* f)
      : filename_(fname), file_(f) {}

  ~TransactionalPosixWritableFile() override {
    if (file_ != nullptr) {
      // Ignoring any potential errors
      fclose(file_);
    }
  }

  Status Append(StringPiece data) override {
    size_t r = fwrite(data.data(), 1, data.size(), file_);
    if (r != data.size()) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Close() override {
    if (file_ == nullptr) {
      return IOError(filename_, EBADF);
    }
    Status result;
    if (fclose(file_) != 0) {
      result = IOError(filename_, errno);
    }
    file_ = nullptr;
    return result;
  }

  Status Flush() override {
    if (fflush(file_) != 0) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Name(StringPiece* result) const override {
    *result = filename_;
    return Status::OK();
  }

  Status Sync() override {
    Status s;
    if (fflush(file_) != 0) {
      s = IOError(filename_, errno);
    }
    return s;
  }

  Status Tell(int64* position) override {
    Status s;
    *position = ftell(file_);

    if (*position == -1) {
      s = IOError(filename_, errno);
    }

    return s;
  }
};

class TransactionalPosixReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  TransactionalPosixReadOnlyMemoryRegion(const void* address, uint64 length)
      : address_(address), length_(length) {}
  ~TransactionalPosixReadOnlyMemoryRegion() override {
    munmap(const_cast<void*>(address_), length_);
  }
  const void* data() override { return address_; }
  uint64 length() override { return length_; }

 private:
  const void* const address_;
  const uint64 length_;
};

Status TransactionalPosixFileSystem::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result,
    TransactionToken* token) {
  string translated_fname = TranslateName(fname);
  VLOG(0)<<"fname "<<fname<<" translated="<<translated_fname;
  if (token) {
    if (token->owner == this) {
      auto ptoken = GetToken(token);
      ptoken->Add(translated_fname);
    } else {
      return errors::InvalidArgument(
          "Given transaction token does not belong to "
          "TransactionalPosixFileSystem!");
    }
  }
  Status s;
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    result->reset(new TransactionalPosixRandomAccessFile(translated_fname, fd));
  }
  return s;
}

Status TransactionalPosixFileSystem::NewWritableFile(
    const string& fname, std::unique_ptr<WritableFile>* result,
    TransactionToken* token) {
  string translated_fname = TranslateName(fname);
  if (token) {
    if (token->owner == this) {
      auto ptoken = GetToken(token);
      ptoken->Add(translated_fname);
    } else {
      return errors::InvalidArgument(
          "Given transaction token does not belong to "
          "TransactionalPosixFileSystem!");
    }
  }

  Status s;
  FILE* f = fopen(translated_fname.c_str(), "w");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new TransactionalPosixWritableFile(translated_fname, f));
  }
  return s;
}

Status TransactionalPosixFileSystem::NewAppendableFile(
    const string& fname, std::unique_ptr<WritableFile>* result,
    TransactionToken* token) {
  string translated_fname = TranslateName(fname);
  if (token) {
    if (token->owner == this) {
      auto ptoken = GetToken(token);
      ptoken->Add(translated_fname);
    } else {
      return errors::InvalidArgument(
          "Given transaction token does not belong to "
          "TransactionalPosixFileSystem!");
    }
  }

  Status s;
  FILE* f = fopen(translated_fname.c_str(), "a");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new TransactionalPosixWritableFile(translated_fname, f));
  }
  return s;
}

Status TransactionalPosixFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result,
    TransactionToken* token) {
  string translated_fname = TranslateName(fname);
  Status s = Status::OK();
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    struct stat st;
    ::fstat(fd, &st);
    const void* address =
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (address == MAP_FAILED) {
      s = IOError(fname, errno);
    } else {
      result->reset(
          new TransactionalPosixReadOnlyMemoryRegion(address, st.st_size));
    }
    if (close(fd) < 0) {
      s = IOError(fname, errno);
    }
  }
  return s;
}

Status TransactionalPosixFileSystem::FileExists(const string& fname,
                                                TransactionToken* token) {
  if (access(TranslateName(fname).c_str(), F_OK) == 0) {
    return Status::OK();
  }
  return errors::NotFound(fname, " not found");
}

Status TransactionalPosixFileSystem::GetChildren(const string& dir,
                                                 std::vector<string>* result,
                                                 TransactionToken* token) {
  string translated_dir = TranslateName(dir);
  result->clear();
  DIR* d = opendir(translated_dir.c_str());
  if (d == nullptr) {
    return IOError(dir, errno);
  }
  struct dirent* entry;
  while ((entry = readdir(d)) != nullptr) {
    StringPiece basename = entry->d_name;
    if ((basename != ".") && (basename != "..")) {
      result->push_back(entry->d_name);
    }
  }
  if (closedir(d) < 0) {
    return IOError(dir, errno);
  }
  return Status::OK();
}

Status TransactionalPosixFileSystem::GetMatchingPaths(
    const string& pattern, std::vector<string>* results,
    TransactionToken* token) {
  if((pattern.length() >= 8) && pattern.substr(0,8)=="trans://"){
    return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
  }else{
    return internal::GetMatchingPaths(this, Env::Default(), pattern.substr(8), results);
  }
}

Status TransactionalPosixFileSystem::DeleteFile(const string& fname,
                                                TransactionToken* token) {
  Status result;
  if (unlink(TranslateName(fname).c_str()) != 0) {
    result = IOError(fname, errno);
  }
  return result;
}

Status TransactionalPosixFileSystem::CreateDir(const string& name,
                                               TransactionToken* token) {
  string translated = TranslateName(name);
  if (translated.empty()) {
    return errors::AlreadyExists(name);
  }
  if (mkdir(translated.c_str(), 0755) != 0) {
    return IOError(name, errno);
  }
  return Status::OK();
}

Status TransactionalPosixFileSystem::DeleteDir(const string& name,
                                               TransactionToken* token) {
  Status result;
  if (rmdir(TranslateName(name).c_str()) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status TransactionalPosixFileSystem::GetFileSize(const string& fname,
                                                 uint64* size,
                                                 TransactionToken* token) {
  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    *size = 0;
    s = IOError(fname, errno);
  } else {
    *size = sbuf.st_size;
  }
  return s;
}

Status TransactionalPosixFileSystem::Stat(const string& fname,
                                          FileStatistics* stats,
                                          TransactionToken* token) {
  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    s = IOError(fname, errno);
  } else {
    stats->length = sbuf.st_size;
    stats->mtime_nsec = sbuf.st_mtime * 1e9;
    stats->is_directory = S_ISDIR(sbuf.st_mode);
  }
  return s;
}

Status TransactionalPosixFileSystem::RenameFile(const string& src,
                                                const string& target,
                                                TransactionToken* token) {
  Status result;
  auto src_name = TranslateName(src);
  auto target_name = TranslateName(target);
  TransactionToken* final_token = token;
  TransactionToken* old_token = nullptr;
  {
    mutex_lock l(transaction_mutex_);
    auto it = transaction_table_.find(src_name);
    if (it != transaction_table_.end()) {
      old_token = it->second;
      transaction_table_.erase(it);
      final_token = (token ? token : old_token);
    }
    transaction_table_.insert({target_name, final_token});
  }
  if (final_token) {
    if (final_token == old_token) {
      GetToken(old_token)->Rename(src_name, target_name);
    } else {
      GetToken(old_token)->Remove(src_name);
      GetToken(final_token)->Add(target_name);
    }
  }
  if (rename(src_name.c_str(), target_name.c_str()) != 0) {
    result = IOError(src, errno);
  }
  return result;
}

Status TransactionalPosixFileSystem::CopyFile(const string& src,
                                              const string& target,
                                              TransactionToken* token) {
  string translated_src = TranslateName(src);
  struct stat sbuf;
  if (stat(translated_src.c_str(), &sbuf) != 0) {
    return IOError(src, errno);
  }
  int src_fd = open(translated_src.c_str(), O_RDONLY);
  if (src_fd < 0) {
    return IOError(src, errno);
  }
  string translated_target = TranslateName(target);
  // O_WRONLY | O_CREAT | O_TRUNC:
  //   Open file for write and if file does not exist, create the file.
  //   If file exists, truncate its size to 0.
  // When creating file, use the same permissions as original
  mode_t mode = sbuf.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
  int target_fd =
      open(translated_target.c_str(), O_WRONLY | O_CREAT | O_TRUNC, mode);
  if (target_fd < 0) {
    close(src_fd);
    return IOError(target, errno);
  }
  int rc = 0;
  off_t offset = 0;
  std::unique_ptr<char[]> buffer(new char[kPosixCopyFileBufferSize]);
  while (offset < sbuf.st_size) {
    // Use uint64 for safe compare SSIZE_MAX
    uint64 chunk = sbuf.st_size - offset;
    if (chunk > SSIZE_MAX) {
      chunk = SSIZE_MAX;
    }
#if defined(__linux__) && !defined(__ANDROID__)
    rc = sendfile(target_fd, src_fd, &offset, static_cast<size_t>(chunk));
#else
    if (chunk > kPosixCopyFileBufferSize) {
      chunk = kPosixCopyFileBufferSize;
    }
    rc = read(src_fd, buffer.get(), static_cast<size_t>(chunk));
    if (rc <= 0) {
      break;
    }
    rc = write(target_fd, buffer.get(), static_cast<size_t>(chunk));
    offset += chunk;
#endif
    if (rc <= 0) {
      break;
    }
  }

  Status result = Status::OK();
  if (rc < 0) {
    result = IOError(target, errno);
  }

  // Keep the error code
  rc = close(target_fd);
  if (rc < 0 && result == Status::OK()) {
    result = IOError(target, errno);
  }
  rc = close(src_fd);
  if (rc < 0 && result == Status::OK()) {
    result = IOError(target, errno);
  }
  return result;
}

Status TransactionalPosixFileSystem::StartTransaction(
    TransactionToken** token) {
  TransactionToken* tok = new TransactionToken;
  tok->owner = this;
  tok->token = new PosixTransactionToken();
  {
    mutex_lock l(transaction_mutex_);
    existing_tokens_.insert(tok);
  }
  *token = tok;
  return Status::OK();
};

/// \brief Adds `path` to transaction in `token`
Status TransactionalPosixFileSystem::AddToTransaction(const string& path,
                                                      TransactionToken* token) {
  if (!token) return Status::OK();
  if (token->owner != this) {
    return errors::InvalidArgument(
        "This token is not for TransactionalPosix filesystem!");
  }
  PosixTransactionToken* tok = GetToken(token);
  auto translated_path = TranslateName(path);
  if (translated_path.empty()) {
    return errors::InvalidArgument("Path can not be empty!");
  }
  tok->Add(translated_path);
  if(VLOG_IS_ON(2)){
    VLOG(2)<<"Adding "<<path<<" to token @"<<(void*)token;
  }
  {
    mutex_lock l(transaction_mutex_);
    transaction_table_.insert({translated_path, token});
  }
  return Status::OK();
};

Status TransactionalPosixFileSystem::FinalizeTransaction(
    TransactionToken* token) {
  return Status::OK();
}

Status TransactionalPosixFileSystem::EndTransaction(TransactionToken* token) {
  if (!token) return Status::OK();
  if (token->owner != this) {
    return errors::InvalidArgument(
        "This token is not for TransactionalPosiz filesystem!");
  }
  if (VLOG_IS_ON(4)) {
    VLOG(1) << "Ending transaction " << DecodeTransaction(token);
  }
  // TODO(sami): implement what transaction means.
  // clear bookkeeping data.
  auto status = FinalizeTransaction(token);
  auto ptoken = GetToken(token);
  {
    mutex_lock l(transaction_mutex_);
    for (const auto& s : ptoken->files) {
      transaction_table_.erase(s);
    }
    existing_tokens_.erase(token);
  }
  delete ptoken;
  delete token;
  return status;
};

/// \brief Get token for `path` or start a new transaction and add it.
Status TransactionalPosixFileSystem::GetTransactionForPath(
    const string& path, TransactionToken** token) {
  if (token == nullptr) {
    return errors::InvalidArgument("Token pointer can't be nullptr");
  }
  *token = nullptr;
  auto translated_path = TranslateName(path);
  if (translated_path.empty()) {
    return errors::InvalidArgument("Path can not be empty!");
  }

  {
    mutex_lock l(transaction_mutex_);
    auto it = transaction_table_.find(translated_path);
    if (it != transaction_table_.end()) {
      *token = it->second;
    }
  }
  return Status::OK();
};

/// \brief Get token for `path` or start a new transaction and add it.
Status TransactionalPosixFileSystem::GetTokenOrStartTransaction(
    const string& path, TransactionToken** token) {
  if (token == nullptr) {
    return errors::InvalidArgument("Token pointer can't be nullptr");
  }
  *token = nullptr;
  auto translated_path = TranslateName(path);
  if (translated_path.empty()) {
    return errors::InvalidArgument("Path can not be empty!");
  }

  {
    mutex_lock l(transaction_mutex_);
    auto it = transaction_table_.find(translated_path);
    if (it != transaction_table_.end()) {
      *token = it->second;
      return Status::OK();
    }
    TransactionToken* tok = new TransactionToken();
    tok->owner = this;
    PosixTransactionToken* pt = new PosixTransactionToken();
    pt->Add(translated_path);
    tok->token = pt;
    transaction_table_.insert({translated_path, tok});
    *token = tok;
  }
  if(VLOG_IS_ON(2)){
    VLOG(2)<<"Adding "<<path<<" to token @"<<(void*)*token;
  }

  return Status::OK();
};

/// \brief Decodes transaction
string TransactionalPosixFileSystem::DecodeTransaction(
    const TransactionToken* token) {
  if (!token) return "Token is nullptr";
  if (token->owner != this) {
    return "This token does not belong to TransactionalPosixFileSystem!";
  }
  auto p = GetToken(token);
  std::stringstream oss;
  oss << "Following files belong to this transaction @" << (void*)token
      << std::endl;
  for (const auto& f : p->files) {
    oss << "  " << f << std::endl;
  }
  return oss.str();
}

TransactionalPosixFileSystem::~TransactionalPosixFileSystem() {
  for (auto& token : existing_tokens_) {
    LOG(WARNING) << "Following transactions was not finalized "
                 << DecodeTransaction(token);
    auto s = FinalizeTransaction(token);
    auto ptoken = GetToken(token);
    if (!s.ok()) {
      LOG(ERROR) << "Failure while finalizing transaction " << s;
    }
    delete ptoken;
    delete token;
  }
}
REGISTER_FILE_SYSTEM("trans", TransactionalPosixFileSystem);
}  // namespace tensorflow
