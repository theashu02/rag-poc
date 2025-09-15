import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { getUserFiles, deleteUserFile } from '@/lib/ApiStore/actions/uploadaction';

export interface UserFile {
  name: string;
  size: number | string;
  createdAt: string;
  type?: string;
  id?: string;
  url?: string;
}

interface FileState {
  files: UserFile[];
  isLoading: boolean;
  error: string | null;
}

const initialState: FileState = {
  files: [],
  isLoading: false,
  error: null,
};

export const fetchUserFiles = createAsyncThunk(
  'files/fetchUserFiles',
  async (userId: string, { rejectWithValue }) => {
    try {
      const result = await getUserFiles(userId);
      if (result.success) {
        return result.success;
      } else {
        return rejectWithValue(result.failure || 'Unknown error occurred');
      }
    } catch (err) {
      return rejectWithValue('Failed to load files. Please try again.');
    }
  }
);

export const removeUserFile = createAsyncThunk(
  'files/deleteUserFile',
  async ({ userId, fileName }: { userId: string; fileName: string }, { rejectWithValue }) => {
    try {
      const result = await deleteUserFile(userId, fileName);
      if (result.success) {
        return fileName;
      } else {
        return rejectWithValue(result.failure || 'Failed to delete file');
      }
    } catch (err) {
      return rejectWithValue('Failed to delete file. Please try again.');
    }
  }
);

const fileSlice = createSlice({
  name: 'files',
  initialState,
  reducers: {
    addFile: (state, action: PayloadAction<UserFile>) => {
      state.files.unshift(action.payload);
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUserFiles.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchUserFiles.fulfilled, (state, action: PayloadAction<UserFile[]>) => {
        state.isLoading = false;
        state.files = action.payload;
      })
      .addCase(fetchUserFiles.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      .addCase(removeUserFile.fulfilled, (state, action: PayloadAction<string>) => {
        state.files = state.files.filter((file) => file.name !== action.payload);
      });
  },
});

export const { addFile } = fileSlice.actions;
export default fileSlice.reducer;
