/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_statistics.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/python/lib/core/pybind11_absl.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace {
namespace py = pybind11;

PYBIND11_MODULE(_pywrap_file_io, m) {
  using tensorflow::TransactionToken;
  py::class_<TransactionToken>(m, "TransactionToken");
  m.def(
      "FileExists",
      [](const std::string& filename, TransactionToken* token) {
        tensorflow::Status status;
        {
          py::gil_scoped_release release;
          status = tensorflow::Env::Default()->FileExists(filename, token);
        }
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("filename"), py::arg("token") = (TransactionToken*)nullptr);
  m.def(
      "DeleteFile",
      [](const std::string& filename, TransactionToken* token) {
        py::gil_scoped_release release;
        tensorflow::Status status =
            tensorflow::Env::Default()->DeleteFile(filename, token);
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("filename"), py::arg("token") = (TransactionToken*)nullptr);
  m.def(
      "ReadFileToString",
      [](const std::string& filename, TransactionToken* token) {
        std::string data;
        py::gil_scoped_release release;
        const auto status =
            ReadFileToString(tensorflow::Env::Default(), filename, &data, token);
        pybind11::gil_scoped_acquire acquire;
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
        return py::bytes(data);
      },
      py::arg("filename"),
      py::arg("token") = (TransactionToken*)nullptr);
  m.def(
      "WriteStringToFile",
      [](const std::string& filename, tensorflow::StringPiece data,
         TransactionToken* token) {
        py::gil_scoped_release release;
        const auto status =
            WriteStringToFile(tensorflow::Env::Default(), filename, data, token);
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("filename"), py::arg("data"),
      py::arg("token") = (TransactionToken*)nullptr);
  m.def(
      "GetChildren",
      [](const std::string& dirname, TransactionToken* token) {
        std::vector<std::string> results;
        py::gil_scoped_release release;
        const auto status =
            tensorflow::Env::Default()->GetChildren(dirname, &results, token);
        pybind11::gil_scoped_acquire acquire;
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
        return results;
      },
      py::arg("dirname"), py::arg("token") = (TransactionToken*)nullptr);
  m.def(
      "GetMatchingFiles",
      [](const std::string& pattern, TransactionToken* token) {
        std::vector<std::string> results;
        py::gil_scoped_release release;
        const auto status =
            tensorflow::Env::Default()->GetMatchingPaths(pattern, &results, token);
        pybind11::gil_scoped_acquire acquire;
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
        return results;
      },
      py::arg("pattern"), py::arg("token") = (TransactionToken*)nullptr);
  m.def(
      "CreateDir",
      [](const std::string& dirname, TransactionToken* token) {
        py::gil_scoped_release release;
        const auto status = tensorflow::Env::Default()->CreateDir(dirname, token);
        if (tensorflow::errors::IsAlreadyExists(status)) {
          return;
        }
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("dirname"), py::arg("token") = (TransactionToken*)nullptr);
  m.def(
      "RecursivelyCreateDir",
      [](const std::string& dirname, TransactionToken* token) {
        py::gil_scoped_release release;
        const auto status =
            tensorflow::Env::Default()->RecursivelyCreateDir(dirname, token);
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("dirname"), py::arg("token") = (TransactionToken*)nullptr);
  m.def(
      "CopyFile",
      [](const std::string& src, const std::string& target, bool overwrite,
         TransactionToken* token) {
        py::gil_scoped_release release;
        auto* env = tensorflow::Env::Default();
        tensorflow::Status status;
        if (!overwrite && env->FileExists(target, token).ok()) {
          status = tensorflow::errors::AlreadyExists("file already exists");
        } else {
          status = env->CopyFile(src, target, token);
        }
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("src"), py::arg("target"), py::arg("overwrite"),
      py::arg("token") = (TransactionToken*)nullptr);
  m.def(
      "RenameFile",
      [](const std::string& src, const std::string& target, bool overwrite,
         TransactionToken* token) {
        py::gil_scoped_release release;
        auto* env = tensorflow::Env::Default();
        tensorflow::Status status;
        if (!overwrite && env->FileExists(target, token).ok()) {
          status = tensorflow::errors::AlreadyExists("file already exists");
        } else {
          status = env->RenameFile(src, target, token);
        }
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("src"), py::arg("target"), py::arg("overwrite"),
      py::arg("token") = (TransactionToken*)nullptr);
  m.def(
      "DeleteRecursively",
      [](const std::string& dirname, TransactionToken* token) {
        py::gil_scoped_release release;
        tensorflow::int64 undeleted_files;
        tensorflow::int64 undeleted_dirs;
        auto status = tensorflow::Env::Default()->DeleteRecursively(
            dirname, &undeleted_files, &undeleted_dirs, token);
        if (status.ok() && (undeleted_files > 0 || undeleted_dirs > 0)) {
          status = tensorflow::errors::PermissionDenied(
              "could not fully delete dir");
        }
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
      },
      py::arg("dirname"), py::arg("token") = (TransactionToken*)nullptr);
  m.def(
      "IsDirectory",
      [](const std::string& dirname, TransactionToken* token) {
        py::gil_scoped_release release;
        const auto status = tensorflow::Env::Default()->IsDirectory(dirname, token);
        // FAILED_PRECONDITION response means path exists but isn't a dir.
        if (tensorflow::errors::IsFailedPrecondition(status)) {
          return false;
        }

        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
        return true;
      },
      py::arg("dirname"), py::arg("token") = (TransactionToken*)nullptr);
  m.def("HasAtomicMove", [](const std::string& path) {
    py::gil_scoped_release release;
    bool has_atomic_move;
    const auto status =
        tensorflow::Env::Default()->HasAtomicMove(path, &has_atomic_move);
    tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
    return has_atomic_move;
  });

  py::class_<tensorflow::FileStatistics>(m, "FileStatistics")
      .def_readonly("length", &tensorflow::FileStatistics::length)
      .def_readonly("mtime_nsec", &tensorflow::FileStatistics::mtime_nsec)
      .def_readonly("is_directory", &tensorflow::FileStatistics::is_directory);

  m.def(
      "Stat",
      [](const std::string& filename, TransactionToken* token) {
        py::gil_scoped_release release;
        std::unique_ptr<tensorflow::FileStatistics> self(
            new tensorflow::FileStatistics);
        const auto status =
            tensorflow::Env::Default()->Stat(filename, self.get(), token);
        py::gil_scoped_acquire acquire;
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
        return self.release();
      },
      py::arg("filename"), py::arg("token") = (TransactionToken*)nullptr);

  m.def("StartTransaction",
        [](const std::string& filename) -> TransactionToken* {
          TransactionToken* self=nullptr;
          const auto status =
              tensorflow::Env::Default()->StartTransaction(filename, &self);
          tensorflow::MaybeRaiseRegisteredFromStatus(status);
          return self;
        });

  m.def("EndTransaction", [](TransactionToken* token) {
    const auto status = tensorflow::Env::Default()->EndTransaction(token);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
  });

  using tensorflow::WritableFile;
  py::class_<WritableFile>(m, "WritableFile")
      .def(py::init([](const std::string& filename, const std::string& mode) {
        py::gil_scoped_release release;
        auto* env = tensorflow::Env::Default();
        std::unique_ptr<WritableFile> self;
        const auto status = mode.find("a") == std::string::npos
                                ? env->NewWritableFile(filename, &self)
                                : env->NewAppendableFile(filename, &self);
        py::gil_scoped_acquire acquire;
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
        return self.release();
      }))
      .def("append",
           [](WritableFile* self, tensorflow::StringPiece data) {
             const auto status = self->Append(data);
             tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
           })
      // TODO(slebedev): Make WritableFile::Tell const and change self
      // to be a reference.
      .def("tell",
           [](WritableFile* self) {
             tensorflow::int64 pos = -1;
             py::gil_scoped_release release;
             const auto status = self->Tell(&pos);
             tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
             return pos;
           })
      .def("flush",
           [](WritableFile* self) {
             py::gil_scoped_release release;
             tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(self->Flush());
           })
      .def("close", [](WritableFile* self) {
        py::gil_scoped_release release;
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(self->Close());
      });

  using tensorflow::io::BufferedInputStream;
  py::class_<BufferedInputStream>(m, "BufferedInputStream")
      .def(py::init([](const std::string& filename, size_t buffer_size) {
        py::gil_scoped_release release;
        std::unique_ptr<tensorflow::RandomAccessFile> file;
        const auto status =
            tensorflow::Env::Default()->NewRandomAccessFile(filename, &file);
        tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
        std::unique_ptr<tensorflow::io::RandomAccessInputStream> input_stream(
            new tensorflow::io::RandomAccessInputStream(file.release(),
                                                        /*owns_file=*/true));
        py::gil_scoped_acquire acquire;
        return new BufferedInputStream(input_stream.release(), buffer_size,
                                       /*owns_input_stream=*/true);
      }))
      .def("read",
           [](BufferedInputStream* self, tensorflow::int64 bytes_to_read) {
             py::gil_scoped_release release;
             tensorflow::tstring result;
             const auto status = self->ReadNBytes(bytes_to_read, &result);
             if (!status.ok() && !tensorflow::errors::IsOutOfRange(status)) {
               result.clear();
               tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(status);
             }
             py::gil_scoped_acquire acquire;
             return py::bytes(result);
           })
      .def("readline",
           [](BufferedInputStream* self) {
             py::gil_scoped_release release;
             auto output = self->ReadLineAsString();
             py::gil_scoped_acquire acquire;
             return py::bytes(output);
           })
      .def("seek",
           [](BufferedInputStream* self, tensorflow::int64 pos) {
             py::gil_scoped_release release;
             tensorflow::MaybeRaiseRegisteredFromStatusWithGIL(self->Seek(pos));
           })
      .def("tell", [](BufferedInputStream* self) {
        py::gil_scoped_release release;
        return self->Tell();
      });
}
}  // namespace
