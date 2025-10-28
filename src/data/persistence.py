"""data persistence layer for storing fetched data in postgresql."""

import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert

from src.data.database import get_database
from src.data.schema import GOESXRayFlux, SolarRegion, FlareEvent, DataIngestionLog

logger = logging.getLogger(__name__)


class DataPersister:
    """handles persistence of fetched data to database."""
    
    def __init__(self):
        self.db = get_database()
    
    def save_xray_flux(self, df: pd.DataFrame, source_name: str = "noaa_goes_xrs") -> dict:
        """
        save goes x-ray flux data to database.
        
        args:
            df: dataframe with columns: timestamp, flux_short, flux_long, satellite
            source_name: name of data source for logging
            
        returns:
            dict with save statistics
        """
        start_time = datetime.utcnow()
        stats = {
            'records_fetched': len(df),
            'records_inserted': 0,
            'records_updated': 0,
            'status': 'success',
            'error_message': None
        }
        
        try:
            with self.db.get_session() as session:
                for _, row in df.iterrows():
                    # prepare data
                    data = {
                        'timestamp': row['timestamp'],
                        'flux_short': row.get('flux_short'),
                        'flux_long': row.get('flux_long'),
                        'satellite': row.get('satellite', 'unknown'),
                        'data_quality': 'good' if pd.notna(row.get('flux_long')) else 'missing'
                    }
                    
                    # upsert (insert or update on conflict)
                    stmt = insert(GOESXRayFlux).values(**data)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['timestamp'],
                        set_={
                            'flux_short': stmt.excluded.flux_short,
                            'flux_long': stmt.excluded.flux_long,
                            'satellite': stmt.excluded.satellite,
                            'data_quality': stmt.excluded.data_quality
                        }
                    )
                    
                    result = session.execute(stmt)
                    
                    # track inserts vs updates (postgresql returns 0 for updates)
                    if result.rowcount > 0:
                        stats['records_inserted'] += 1
                
                session.commit()
                
            logger.info(f"saved {stats['records_inserted']} xray flux records")
            
        except Exception as e:
            stats['status'] = 'failure'
            stats['error_message'] = str(e)
            logger.error(f"failed to save xray flux data: {e}")
        
        finally:
            # log ingestion
            duration = (datetime.utcnow() - start_time).total_seconds()
            self._log_ingestion(
                source_name=source_name,
                data_start=df['timestamp'].min() if len(df) > 0 else None,
                data_end=df['timestamp'].max() if len(df) > 0 else None,
                duration=duration,
                **stats
            )
        
        return stats
    
    def save_solar_regions(self, df: pd.DataFrame, source_name: str = "noaa_solar_regions") -> dict:
        """
        save solar region data to database.
        
        args:
            df: dataframe with solar region data
            source_name: name of data source for logging
            
        returns:
            dict with save statistics
        """
        start_time = datetime.utcnow()
        stats = {
            'records_fetched': len(df),
            'records_inserted': 0,
            'records_updated': 0,
            'status': 'success',
            'error_message': None
        }
        
        try:
            with self.db.get_session() as session:
                for _, row in df.iterrows():
                    region = SolarRegion(
                        timestamp=row.get('timestamp', datetime.utcnow()),
                        region_number=row.get('region_number'),
                        latitude=row.get('latitude'),
                        longitude=row.get('longitude'),
                        mcintosh_class=row.get('mcintosh_class'),
                        mount_wilson_class=row.get('mount_wilson_class'),
                        area=row.get('area'),
                        num_sunspots=row.get('num_sunspots'),
                        magnetic_type=row.get('magnetic_type')
                    )
                    
                    session.add(region)
                    stats['records_inserted'] += 1
                
                session.commit()
                
            logger.info(f"saved {stats['records_inserted']} solar region records")
            
        except Exception as e:
            stats['status'] = 'failure'
            stats['error_message'] = str(e)
            logger.error(f"failed to save solar region data: {e}")
        
        finally:
            duration = (datetime.utcnow() - start_time).total_seconds()
            self._log_ingestion(
                source_name=source_name,
                data_start=df['timestamp'].min() if len(df) > 0 and 'timestamp' in df else None,
                data_end=df['timestamp'].max() if len(df) > 0 and 'timestamp' in df else None,
                duration=duration,
                **stats
            )
        
        return stats
    
    def get_latest_xray_timestamp(self) -> Optional[datetime]:
        """
        get timestamp of most recent xray flux record.
        
        returns:
            datetime of latest record or none if no data
        """
        try:
            with self.db.get_session() as session:
                result = session.query(func.max(GOESXRayFlux.timestamp)).scalar()
                return result
        except Exception as e:
            logger.error(f"failed to get latest timestamp: {e}")
            return None
    
    def get_xray_flux_range(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        retrieve xray flux data for a time range.
        
        args:
            start_time: start of time range
            end_time: end of time range
            
        returns:
            dataframe with flux data
        """
        try:
            with self.db.get_session() as session:
                query = session.query(GOESXRayFlux).filter(
                    GOESXRayFlux.timestamp >= start_time,
                    GOESXRayFlux.timestamp <= end_time
                ).order_by(GOESXRayFlux.timestamp)
                
                records = query.all()
                
                # convert to dataframe
                data = [{
                    'timestamp': r.timestamp,
                    'flux_short': r.flux_short,
                    'flux_long': r.flux_long,
                    'satellite': r.satellite,
                    'data_quality': r.data_quality
                } for r in records]
                
                return pd.DataFrame(data)
                
        except Exception as e:
            logger.error(f"failed to retrieve xray flux data: {e}")
            return pd.DataFrame()
    
    def _log_ingestion(self, source_name: str, data_start: Optional[datetime],
                       data_end: Optional[datetime], duration: float,
                       records_fetched: int, records_inserted: int,
                       records_updated: int, status: str,
                       error_message: Optional[str]):
        """log data ingestion run."""
        try:
            with self.db.get_session() as session:
                log = DataIngestionLog(
                    source_name=source_name,
                    status=status,
                    records_fetched=records_fetched,
                    records_inserted=records_inserted,
                    records_updated=records_updated,
                    data_start_time=data_start,
                    data_end_time=data_end,
                    error_message=error_message[:500] if error_message else None,
                    duration_seconds=duration
                )
                session.add(log)
                session.commit()
        except Exception as e:
            logger.error(f"failed to log ingestion: {e}")

