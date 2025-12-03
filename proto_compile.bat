::Directory of protobuf, should be installed alongside grpc
::VCPKG_ROOT is the same as the one you set earlier
::Worst case you can copy the whole path here starting from the vcpkg folder
set protobuf_cmd="%VCPKG_ROOT%\packages\protobuf_x64-windows\tools\protobuf\protoc"
:: Directory where the grpc_cpp_plugin.exe can be found
::Other things are the same as above
set grpc_exe_dir="%VCPKG_ROOT%\installed\x64-windows\tools\grpc\grpc_cpp_plugin.exe"
:: Folder where the source of the proto file
set src="%cd%"
::File to compile
set proto_file="%cd%\dashboard.proto"
:: Folder where we dump the generated files
set dest="%cd%\generated"
:: The compile command itself
%protobuf_cmd% --proto_path=%src% --cpp_out=%dest% --grpc_out=%dest% --plugin=protoc-gen-grpc=%grpc_exe_dir% %proto_file%