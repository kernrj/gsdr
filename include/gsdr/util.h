/*
 * Copyright (C) 2023 Rick Kern <kernrj@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General
 * Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Affero General Public License along with this program.  If not, see
 * <https://www.gnu.org/licenses/>.
 */

#ifndef GSDR_INCLUDE_GSDR_UTIL_H_
#define GSDR_INCLUDE_GSDR_UTIL_H_

#ifdef __cplusplus
#define GSDR_C_LINKAGE extern "C"
#else
#define GSDR_C_LINKAGE
#endif

#ifdef __cplusplus
#define GSDR_NO_EXCEPT noexcept
#else
#define GSDR_NO_EXCEPT
#endif

#endif  // GSDR_INCLUDE_GSDR_UTIL_H_
